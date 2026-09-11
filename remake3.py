#!/usr/bin/env python3
"""Interactive, ROS-free prototype for inspecting candidate planning knowledge.

Run: python remake3.py
Actions and predicates are selected from data/*.yaml using lexical evidence only.
Matching policy is documented above select_catalog_entries().
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

LOGGER = logging.getLogger("remake3")


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


def catalog_terms(name: str, definition: dict) -> set[str]:
    extra = definition.get("synonyms", [])
    if isinstance(extra, str):
        extra = [extra]
    if not isinstance(extra, list) or not all(isinstance(term, str) for term in extra):
        raise ValueError(f"Synonyms for {name} must be a list of strings")
    return {name.replace("_", " "), *extra}


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
# Catalog selection policy (shared by actions and predicates):
# - Only catalog names and optional YAML `synonyms` supply target vocabulary.
# - Underscores in names mean spaces; compound names are kept whole.
# - Exact token sequences, WordNet synonyms and direct hypernyms/hyponyms match.
# - Keep all matching catalog entries; do not choose a sense or rank entries.
# - Actions use verb candidates; predicates use literal instruction phrases and
#   POS-tagged content words. No action-to-predicate or preposition rules.
# Intent descriptions and capabilities are metadata, not matching evidence.
# Type inference and the upstream imperative/role heuristics remain unchanged.
# ---------------------------------------------------------------------------


def select_catalog_entries(analysis: dict, catalog: dict, wordnet, kind: str) -> list[dict]:
    tokens = [token.lower() for token, _ in analysis["tokens"]]
    verb_indexes = {verb["index"] for verb in analysis["verbs"]}
    if kind == "actions":
        words = [(verb["lemma"], "v") for verb in analysis["verbs"]]
    else:
        pos_map = {"NN": "n", "VB": "v", "JJ": "a", "RB": "r"}
        words = [(token, pos_map[tag[:2]]) for token, tag in analysis["tokens"]
                 if tag[:2] in pos_map]

    selected = []
    for name, definition in catalog.items():
        terms = catalog_terms(name, definition)
        reasons = []
        for term in sorted(terms):
            phrase = term.lower().replace("_", " ").split()
            for index in range(len(tokens) - len(phrase) + 1):
                if kind == "actions" and index not in verb_indexes:
                    continue
                if tokens[index:index + len(phrase)] == phrase:
                    reasons.append(f"literal catalog name/alias '{term}' at token {index}")
        if wordnet is not None:
            for word, pos in dict.fromkeys(words):
                match = wordnet_match(word, terms, wordnet, pos)
                if match:
                    reasons.append(match[1])
            # Match compound WordNet lemmas as whole phrases, never by taking
            # just the head word of a compound catalog name.
            max_length = max((len(term.replace("_", " ").split()) for term in terms), default=1)
            for length in range(2, max_length + 1):
                for index in range(len(tokens) - length + 1):
                    if kind == "actions" and index not in verb_indexes:
                        continue
                    phrase = " ".join(tokens[index:index + length])
                    positions = ("v",) if kind == "actions" else ("n", "v", "a", "r")
                    for pos in positions:
                        match = wordnet_match(phrase, terms, wordnet, pos)
                        if match:
                            reasons.append(match[1])
        if reasons:
            reasons = list(dict.fromkeys(reasons))
            selected.append({"name": name, "matching_evidence": reasons})
            log_data(f"Selected {kind}: {name}", reasons)
        else:
            LOGGER.debug("No lexical evidence for %s: %s", kind, name)
    LOGGER.info("%s candidates: %s", kind, [entry["name"] for entry in selected])
    LOGGER.info("All matches retained, including ambiguous senses; no domain disambiguation applied.")
    return selected


def infer_actions(analysis: dict, catalog: dict, wordnet=None) -> list[dict]:
    log_section("5. Match action catalog vocabulary")
    return select_catalog_entries(analysis, catalog, wordnet, "actions")


def infer_predicates(analysis: dict, catalog: dict, wordnet=None) -> list[dict]:
    log_section("6. Match predicate catalog vocabulary independently")
    return select_catalog_entries(analysis, catalog, wordnet, "predicates")


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
    predicates = infer_predicates(analysis, predicate_catalog, resources["wordnet"])
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
    folder_name = f"{slug}_remake3_{timestamp}"
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
