#!/usr/bin/env python3
"""Interactive, ROS-free prototype for inspecting candidate planning knowledge.

Run: python remake2.py
Actions and predicates are selected from data/*.yaml using explicit rules.
Dependencies: pip install nltk pyyaml
Missing NLTK datasets are downloaded on demand. The entity/role bindings are
heuristics, not semantic role labeling. Output is a candidate model, not a
complete executable PDDL domain. This file does not import the existing project.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Debug output: use one logger so every stage is easy to follow.
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("remake2")


def configure_logging() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")


def log_section(title: str) -> None:
    LOGGER.info("\n%s\n%s\n%s", "=" * 65, title, "=" * 65)


def log_data(label: str, value: object) -> None:
    LOGGER.debug("%s:\n%s", label, json.dumps(value, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Resources: load lazily, after asking for the instruction.
# ---------------------------------------------------------------------------


def load_resources() -> dict:
    import nltk
    from nltk.corpus import propbank, verbnet, wordnet
    from nltk.stem import WordNetLemmatizer

    log_section("1. Load linguistic resources")
    lemmatizer = WordNetLemmatizer()
    checks = [
        ("punkt_tab", lambda: nltk.word_tokenize("Move the cup.")),
        ("averaged_perceptron_tagger_eng", lambda: nltk.pos_tag(["move"])),
        ("wordnet", lambda: lemmatizer.lemmatize("moving", "v")),
        ("propbank", lambda: propbank.rolesets()),
        ("verbnet", lambda: verbnet.classids()),
    ]
    for package, check in checks:
        try:
            check()
        except LookupError:
            LOGGER.info("Downloading missing NLTK resource: %s", package)
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Could not download NLTK resource: {package}")
            check()
        LOGGER.debug("Ready: %s", package)
    return {"nltk": nltk, "lemmatizer": lemmatizer,
            "propbank": propbank, "verbnet": verbnet, "wordnet": wordnet}


# ---------------------------------------------------------------------------
# Linguistic analysis: tokens, verb candidates, noun phrases, tentative roles.
# No planning types or predicates are assigned here.
# ---------------------------------------------------------------------------

PREPOSITIONS = {"in", "into", "inside", "within", "on", "onto", "to", "from", "with", "at"}
CONTAINER_PREPOSITIONS = {"in", "into", "inside", "within"}


def analyze_instruction(instruction: str, resources: dict) -> dict:
    log_section("2. Analyze the instruction")
    nltk = resources["nltk"]
    tokens = nltk.word_tokenize(instruction.lower())
    tagged = nltk.pos_tag(tokens)
    log_data("Tokens and part-of-speech tags", tagged)
    verbs = []
    for index, (token, tag) in enumerate(tagged):
        # Imperatives can be tagged as nouns. Explicitly expose this assumption.
        imperative = index == 0 and token.isalpha() and token != "please"
        polite_imperative = index == 1 and tokens[0] == "please" and token.isalpha()
        if tag.startswith("VB") or imperative or polite_imperative:
            verbs.append({"index": index, "text": token,
                          "lemma": resources["lemmatizer"].lemmatize(token, "v")})
    LOGGER.info("Assuming an imperative; the first word after optional 'please' is a verb.")
    verb_indexes = {verb["index"] for verb in verbs}
    entities = []
    index = 0
    while index < len(tagged):
        token, tag = tagged[index]
        if index in verb_indexes or not tag.startswith(("NN", "JJ")):
            index += 1
            continue
        start = index
        phrase = []
        noun = None
        while index < len(tagged) and index not in verb_indexes:
            word, part = tagged[index]
            if not part.startswith(("NN", "JJ")):
                break
            phrase.append(word)
            if part.startswith("NN"):
                noun = word
            index += 1
        if noun is None:
            continue
        preceding = start - 1
        while preceding >= 0 and tagged[preceding][1] in {"DT", "PRP$"}:
            preceding -= 1
        prep = tokens[preceding] if preceding >= 0 and tokens[preceding] in PREPOSITIONS else None
        preceding_verbs = [verb for verb in verbs if verb["index"] < start]
        owner = preceding_verbs[-1]["index"] if preceding_verbs else None
        role = {"from": "Source", "with": "Instrument", "at": "Location"}.get(prep)
        if role is None:
            role = "Destination" if prep else "Theme"
        entities.append({"id": f"entity_{len(entities) + 1}", "text": " ".join(phrase),
                         "head": resources["lemmatizer"].lemmatize(noun, "n"),
                         "verb_index": owner, "role": role, "preposition": prep,
                         "binding_basis": "noun position and nearby preposition (heuristic)"})
    result = {"instruction": instruction, "tokens": tagged, "verbs": verbs,
              "entities": entities, "negated": any(t in {"not", "n't", "never"} for t in tokens)}
    log_data("Shared linguistic representation", result)
    LOGGER.warning("Bindings are heuristic; coordination, pronouns and complex clauses may be misread.")
    return result


# ---------------------------------------------------------------------------
# Evidence collection: retain candidate senses and their original role data.
# Collect once; downstream inference and diagnostics share this evidence.
# ---------------------------------------------------------------------------


def collect_evidence(analysis: dict, resources: dict) -> dict:
    log_section("3. Collect PropBank / VerbNet evidence")
    lemmas = {verb["lemma"] for verb in analysis["verbs"]}
    evidence = {lemma: [] for lemma in lemmas}
    for frame in resources["propbank"].rolesets():
        lemma = frame.attrib["id"].rsplit(".", 1)[0]
        if lemma not in lemmas:
            continue
        roles = []
        for role in frame.findall("roles/role"):
            mappings = []
            for mapping in role.findall("vnrole"):
                class_id = mapping.attrib.get("vncls")
                theta = mapping.attrib.get("vntheta")
                if not class_id or not theta:
                    continue
                try:
                    frames = resources["verbnet"].frames(class_id)
                    semantics = [pred for vn_frame in frames for pred in vn_frame.get("semantics", [])]
                except (ValueError, KeyError) as exc:
                    LOGGER.debug("Unavailable VerbNet class %s: %s", class_id, exc)
                    semantics = []
                mappings.append({"class": class_id, "theta": theta, "semantics": semantics})
            roles.append({"argument": role.attrib.get("n"),
                          "description": role.attrib.get("descr", ""), "verbnet": mappings})
        evidence[lemma].append({"sense": frame.attrib["id"], "name": frame.attrib.get("name"), "roles": roles})
    log_data("Candidate senses and semantic evidence", evidence)
    LOGGER.info("All matching senses are retained; no word-sense disambiguation is performed.")
    return evidence


# ---------------------------------------------------------------------------
# Type inference: candidate declarations with an explanation for each one.
# ---------------------------------------------------------------------------


def infer_types(analysis: dict, evidence: dict) -> list[dict]:
    log_section("4. Infer candidate types")
    reasons = {"object": {"PDDL root type"}, "locatable - object": {"base hierarchy"},
               "location - object": {"base hierarchy"}, "robot - locatable": {"base hierarchy"},
               "item - locatable": {"base hierarchy"}}
    for entity in analysis["entities"]:
        if entity["preposition"] in CONTAINER_PREPOSITIONS:
            reasons.setdefault("container - item", set()).add(f"{entity['text']}: container preposition")
    for lemma, senses in evidence.items():
        for sense in senses:
            for role in sense["roles"]:
                if role["argument"] == "0":
                    continue
                for mapping in role["verbnet"]:
                    if mapping["theta"] not in {"Theme", "Patient", "Destination", "Source", "Location", "Instrument"}:
                        continue
                    for predicate in mapping["semantics"]:
                        mentions_role = any(arg.get("type") == "ThemRole" and
                                            arg.get("value", "").strip() == mapping["theta"]
                                            for arg in predicate.get("arguments", []))
                        if predicate.get("predicate_value") == "motion" and mentions_role:
                            reasons.setdefault("movable - item", set()).add(
                                f"{lemma}/{sense['sense']}: motion involving {mapping['theta']}")
    result = [{"declaration": name, "reasons": sorted(why)} for name, why in sorted(reasons.items())]
    log_data("Types and provenance", result)
    return result


# ---------------------------------------------------------------------------
# Catalog loading: YAML keys are the only allowed action/predicate names.
# Intent descriptions are retained for logging; matching uses explicit rules.
# ---------------------------------------------------------------------------


def load_catalogs(data_dir: str | Path | None = None) -> tuple[dict, dict]:
    import yaml

    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "data"
    catalogs = []
    for kind in ("actions", "predicates"):
        path = root / f"{kind}.yaml"
        catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(catalog, dict) or not all(
            isinstance(name, str) and isinstance(details, dict)
            and isinstance(details.get("intent"), str)
            for name, details in catalog.items()
        ):
            raise ValueError(f"{path} must map names to definitions with an intent string")
        LOGGER.info("Loaded %d %s from %s", len(catalog), kind, path)
        catalogs.append(catalog)
    return tuple(catalogs)


# ---------------------------------------------------------------------------
# WordNet matching: synonyms share a synset; hierarchy matches are direct
# hypernym/hyponym edges only. No shared-ancestor matching (too permissive).
# All senses are candidates, so these are suggestions rather than certainty.
# Optional `synonyms: [alias, ...]` in either YAML catalog adds local names.
# ---------------------------------------------------------------------------


def catalog_terms(name: str, definition: dict, aliases: set[str]) -> set[str]:
    extra = definition.get("synonyms", [])
    if isinstance(extra, str):
        extra = [extra]
    if not isinstance(extra, list) or not all(isinstance(term, str) for term in extra):
        raise ValueError(f"Synonyms for {name} must be a list of strings")
    return {name.replace("_", " "), *aliases, *extra}


def wordnet_match(word: str, terms: set[str], wordnet, pos: str) -> tuple[int, str] | None:
    """Return the strongest lexical relation and its inspectable synset evidence."""
    normalized = word.lower().replace(" ", "_")
    sources = wordnet.synsets(normalized, pos=pos)
    best = None
    for term in sorted(terms):
        target = term.lower().replace(" ", "_")
        if normalized == target:
            return 4, f"exact alias '{term}'"
        targets = set(wordnet.synsets(target, pos=pos))
        for source in sources:
            if source in targets:
                return 3, f"synonym '{word}' ~ '{term}' in {source.name()}"
            for relation, neighbors in (("hypernym", source.hypernyms()),
                                        ("hyponym", source.hyponyms())):
                for neighbor in neighbors:
                    if neighbor in targets and best is None:
                        best = (2, f"'{term}' is a direct {relation} of '{word}': "
                                   f"{source.name()} -> {neighbor.name()}")
    return best


# ---------------------------------------------------------------------------
# Action selection: synonyms plus local context disambiguate catalog actions.
# These rules propose relevant vocabulary, not a plan or an execution order.
# Add aliases here when introducing catalog entries with unfamiliar synonyms.
# ---------------------------------------------------------------------------

ACTION_ALIASES = {
    "drive": {"drive", "navigate", "go", "travel"},
    "move": {"move", "reach"},
    "pick": {"pick", "grasp", "grab", "hold", "lift", "take"},
    "place": {"place", "put", "set"},
    "place_in_container": {"bag", "pack", "insert"},
    "stack": {"stack"}, "unstack": {"unstack"},
    "open": {"open"}, "close": {"close", "shut"},
    "push": {"push"}, "pull": {"pull"}, "pour": {"pour"},
    "look_at": {"look", "inspect", "observe", "watch"},
    "clean": {"clean", "wash", "wipe", "scrub", "dust", "sweep"},
    "fold": {"fold"}, "unfold": {"unfold"},
}


def infer_actions(analysis: dict, catalog: dict, wordnet=None) -> list[dict]:
    log_section("5. Select catalog actions")
    selected = {}
    verbs = analysis["verbs"]
    for position, verb in enumerate(verbs):
        end = verbs[position + 1]["index"] if position + 1 < len(verbs) else len(analysis["tokens"])
        words = []
        for token, _ in analysis["tokens"][verb["index"]:end]:
            if token in {".", ";", "!", "?"}:
                break
            words.append(token)
        lemma = verb["lemma"]
        candidates = {name for name, aliases in ACTION_ALIASES.items() if lemma in aliases}
        if lemma in catalog:
            candidates.add(lemma)
        # A destination inside a container refines generic placement/movement.
        if lemma in {"put", "place", "set", "move"} and set(words) & CONTAINER_PREPOSITIONS:
            candidates = {"place_in_container"}
        elif lemma == "move" and set(words) & {"on", "onto"}:
            candidates = {"place"}
        # Explicit contextual rules take priority over broader lexical senses.
        if not candidates and wordnet is not None:
            matches = {}
            for name, definition in catalog.items():
                terms = catalog_terms(name, definition, ACTION_ALIASES.get(name, set()))
                match = wordnet_match(lemma, terms, wordnet, "v")
                if match:
                    matches[name] = match
            if matches:
                strongest = max(score for score, _ in matches.values())
                candidates = {name for name, (score, _) in matches.items() if score == strongest}
                for name in sorted(candidates):
                    LOGGER.info("WordNet action %s: %s", name, matches[name][1])
        for name in sorted(candidates):
            if name not in catalog:
                LOGGER.debug("Skip %s: absent from action catalog", name)
                continue
            selected[name] = {"name": name}
            LOGGER.info("Selected action %s: verb '%s', context '%s'; %s", name,
                        lemma, " ".join(words), catalog[name]["intent"])
        if not candidates:
            LOGGER.debug("No action rule for verb '%s'", lemma)
    LOGGER.info("Action selection is heuristic; robot capabilities are not checked.")
    return list(selected.values())


# ---------------------------------------------------------------------------
# Predicate selection: explicit phrases and state relations relevant to actions.
# Selecting 'open' for 'close' means the state variable is relevant, not true.
# ---------------------------------------------------------------------------

ACTION_PREDICATES = {
    "drive": {"at"}, "move": {"at"},
    "pick": {"holding", "robot_free"},
    "place": {"on_top", "holding", "robot_free"},
    "place_in_container": {"in", "holding", "robot_free"},
    "stack": {"on_top", "holding", "robot_free"},
    "unstack": {"on_top", "holding", "robot_free"},
    "open": {"open"}, "close": {"open"},
    "push": {"at", "touching"}, "pull": {"at", "touching"},
    "pour": {"in", "holding"}, "clean": {"covered"},
    "fold": {"folded", "unfolded"}, "unfold": {"folded", "unfolded"},
}
PREDICATE_PHRASES = {
    "at": {"at", "to", "from"}, "in": CONTAINER_PREPOSITIONS,
    "on_top": {"on", "onto", "on top of"},
    "next_to": {"next to", "beside", "adjacent"},
    "under": {"under", "underneath", "below"},
    "touching": {"touching", "in contact with"},
    "attached": {"attached", "connected", "fastened"},
    "draped": {"draped"}, "holding": {"holding"},
    "robot_free": {"empty handed", "empty-handed"},
    "covered": {"dirty", "dusty", "stained", "residue", "covered"},
    "open": {"open", "closed"}, "cooked": {"cooked", "raw"},
    "frozen": {"frozen", "thawed"}, "folded": {"folded"},
    "unfolded": {"unfolded"}, "toggled_on": {"switched on", "switched off", "turned on", "turned off"},
    "hot": {"hot"}, "broken": {"broken"}, "on_fire": {"on fire", "burning"},
}


def infer_predicates(analysis: dict, actions: list[dict], catalog: dict, wordnet=None) -> list[dict]:
    log_section("6. Select catalog predicates")
    text = analysis["instruction"].lower()
    reasons = {}
    for action in actions:
        for name in ACTION_PREDICATES.get(action["name"], set()):
            reasons.setdefault(name, []).append(f"relevant to action {action['name']}")
    for name in catalog:
        phrases = catalog_terms(name, catalog[name], PREDICATE_PHRASES.get(name, set()))
        for phrase in phrases:
            if re.search(r"\b" + re.escape(phrase) + r"\b", text):
                # Avoid interpreting the 'on' in 'on fire' as a support relation.
                if name == "on_top" and phrase == "on":
                    if not re.search(r"\bon\b(?!\s+(?:fire|top\b))", text):
                        continue
                reasons.setdefault(name, []).append(f"instruction phrase '{phrase}'")
    if wordnet is not None:
        # State adjectives and verbs are useful here; object nouns must not
        # become state predicates merely because they have unrelated verb senses.
        for token, tag in analysis["tokens"]:
            pos = "a" if tag.startswith("JJ") else "v" if tag.startswith("VB") else None
            if pos is None:
                continue
            matches = {}
            for name, definition in catalog.items():
                terms = catalog_terms(name, definition, PREDICATE_PHRASES.get(name, set()))
                match = wordnet_match(token, terms, wordnet, pos)
                if match:
                    matches[name] = match
            if matches:
                strongest = max(score for score, _ in matches.values())
                for name, (score, reason) in matches.items():
                    if score == strongest:
                        reasons.setdefault(name, []).append("WordNet: " + reason)
    selected = []
    for name in catalog:
        if name in reasons:
            selected.append({"name": name})
            LOGGER.info("Selected predicate %s: %s; %s", name,
                        "; ".join(reasons[name]), catalog[name]["intent"])
    LOGGER.info("Predicates are relevant vocabulary, not assertions of current truth.")
    return selected


# ---------------------------------------------------------------------------
# Assembly: enforce catalog membership and preserve the original type inference.
# ---------------------------------------------------------------------------


def assemble_result(analysis: dict, evidence: dict, types: list[dict],
                    predicates: list[dict], actions: list[dict],
                    action_catalog: dict, predicate_catalog: dict) -> dict:
    log_section("7. Check catalog membership")
    for entries, catalog in ((actions, action_catalog), (predicates, predicate_catalog)):
        for entry in entries:
            if entry["name"] not in catalog:
                raise ValueError(f"Selected name absent from catalog: {entry['name']}")
    return {"instruction": analysis["instruction"], "analysis": analysis,
            "evidence": evidence, "types": types, "predicates": predicates,
            "actions": actions, "complete_pddl_domain": False}


def process_instruction(instruction: str, data_dir: str | Path | None = None,
                        tasks_dir: str | Path | None = None) -> dict:
    """Analyze one instruction, select catalog vocabulary, and save the task."""
    if not instruction.strip():
        raise ValueError("Instruction must not be empty.")
    action_catalog, predicate_catalog = load_catalogs(data_dir)
    resources = load_resources()
    analysis = analyze_instruction(instruction, resources)
    evidence = collect_evidence(analysis, resources)
    types = infer_types(analysis, evidence)
    actions = infer_actions(analysis, action_catalog, resources["wordnet"])
    predicates = infer_predicates(analysis, actions, predicate_catalog, resources["wordnet"])
    result = assemble_result(analysis, evidence, types, predicates, actions,
                             action_catalog, predicate_catalog)
    save_task(result, tasks_dir)
    return result


# ---------------------------------------------------------------------------
# Task files: one unique instruction folder, with readable YAML candidates.
# Timestamps track runs; numeric suffixes handle runs within the same second.
# ---------------------------------------------------------------------------


def save_task(result: dict, tasks_dir: str | Path | None = None) -> Path:
    """Save the instruction, names and type declarations; return the folder."""
    import yaml

    log_section("8. Save task files")
    root = Path(tasks_dir) if tasks_dir is not None else Path(__file__).resolve().parent / "tasks"
    root.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result["instruction"].lower()).strip("-")[:80].rstrip("-") or "instruction"
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S%z")
    folder_name = f"{slug}_remake2_{timestamp}"
    folder = root / folder_name
    suffix = 2
    while True:
        try:
            folder.mkdir()
            break
        except FileExistsError:
            folder = root / f"{folder_name}-{suffix}"
            suffix += 1

    instruction_file = folder / "instructions.txt"
    instruction_file.write_text(result["instruction"] + "\n", encoding="utf-8")
    LOGGER.debug("Wrote %s", instruction_file)
    for name in ("predicates", "types", "actions"):
        path = folder / f"{name}.yaml"
        field = "declaration" if name == "types" else "name"
        values = list(dict.fromkeys(entry[field] for entry in result[name]))
        path.write_text(yaml.safe_dump({name: values}, sort_keys=False, allow_unicode=True), encoding="utf-8")
        LOGGER.debug("Wrote %s", path)
    LOGGER.info("Task saved to %s", folder)
    return folder


# ---------------------------------------------------------------------------
# Standalone interactive entry point: ask first, then execute the pipeline.
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging()
    try:
        instruction = input("Enter an instruction: ").strip()
        result = process_instruction(instruction)
        log_section("9. Final candidate model (full evidence appears above)")
        print(json.dumps({key: result[key] for key in
                          ("instruction", "types", "predicates", "actions", "complete_pddl_domain")}, indent=2))
    except (EOFError, KeyboardInterrupt):
        print("\nStopped.")
    except Exception:
        LOGGER.exception("Extraction failed. See the traceback for the failing stage.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
