#!/usr/bin/env python3
"""Interactive, ROS-free prototype for inspecting candidate planning knowledge.

Run: python remake.py
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

LOGGER = logging.getLogger("remake")


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
    from nltk.corpus import propbank, verbnet
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
            "propbank": propbank, "verbnet": verbnet}


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
# Predicate inference: explicit modeling templates, not corpus facts.
# ---------------------------------------------------------------------------


def infer_predicates(analysis: dict) -> list[dict]:
    log_section("5. Infer candidate predicates")
    predicates = {}
    for entity in analysis["entities"]:
        prep = entity["preposition"]
        name = "in" if prep in CONTAINER_PREPOSITIONS else "on" if prep in {"on", "onto"} else "at" if prep in {"to", "at", "from"} else None
        if name:
            target_type = "container" if name == "in" else "object"
            predicates[name] = {"name": name,
                                "declaration": f"({name} ?item - item ?target - {target_type})",
                                "basis": f"Modeling template for preposition '{prep}'"}
    result = list(predicates.values())
    log_data("Candidate predicates", result)
    return result


# ---------------------------------------------------------------------------
# Action inference: propose parameters and limited effects. Unknown conditions
# remain explicit. Never turn these incomplete candidates into runnable PDDL.
# ---------------------------------------------------------------------------

TRANSFER_VERBS = {"put", "place", "move", "insert", "pour", "take", "carry"}


def infer_actions(analysis: dict, predicates: list[dict]) -> list[dict]:
    log_section("6. Infer candidate actions")
    actions = []
    predicate_names = {predicate["name"] for predicate in predicates}
    for verb in analysis["verbs"]:
        entities = [entity for entity in analysis["entities"] if entity["verb_index"] == verb["index"]]
        parameters = [{"variable": f"?{entity['id']}", "type": "container" if entity["preposition"] in CONTAINER_PREPOSITIONS else "item" if entity["role"] == "Theme" else "object",
                       "observed_entity": entity["text"], "role": entity["role"]} for entity in entities]
        effects = []
        themes = [entity for entity in entities if entity["role"] == "Theme"]
        if verb["lemma"] in TRANSFER_VERBS and len(themes) == 1 and not analysis["negated"]:
            for target in entities:
                prep = target["preposition"]
                relation = "in" if prep in CONTAINER_PREPOSITIONS else "on" if prep in {"on", "onto"} else "at" if prep == "to" else None
                if relation in predicate_names:
                    effects.append(f"({relation} ?{themes[0]['id']} ?{target['id']})")
        actions.append({"name": verb["lemma"], "occurrence": verb["index"], "parameters": parameters,
                        "preconditions": None, "candidate_add_effects": effects,
                        "delete_effects": None, "status": "incomplete candidate",
                        "assumptions": ["Positional entity bindings are tentative.",
                                        "Transfer verbs may establish the stated destination relation.",
                                        "Preconditions and delete effects require domain knowledge."]})
    if analysis["negated"]:
        LOGGER.warning("Negation detected: effect proposals are suppressed for this instruction.")
    log_data("Candidate actions (null means unknown)", actions)
    return actions


# ---------------------------------------------------------------------------
# Assembly and basic consistency checks (not full PDDL validation).
# ---------------------------------------------------------------------------


def assemble_result(analysis: dict, evidence: dict, types: list[dict],
                    predicates: list[dict], actions: list[dict]) -> dict:
    log_section("7. Assemble and check candidates")
    declared = {entry["declaration"].split(" - ")[0] for entry in types}
    for action in actions:
        for parameter in action["parameters"]:
            if parameter["type"] not in declared:
                raise ValueError(f"Undeclared parameter type: {parameter['type']}")
        bound = {parameter["variable"] for parameter in action["parameters"]}
        for effect in action["candidate_add_effects"]:
            if not set(re.findall(r"\?[\w-]+", effect)) <= bound:
                raise ValueError(f"Unbound effect variable: {effect}")
    LOGGER.info("Parameter types and effect variable bindings checked.")
    return {"instruction": analysis["instruction"], "analysis": analysis,
            "evidence": evidence, "types": types, "predicates": predicates,
            "actions": actions, "complete_pddl_domain": False}


def process_instruction(instruction: str) -> dict:
    """Public entry point; can also be imported without prompting or downloading."""
    if not instruction.strip():
        raise ValueError("Instruction must not be empty.")
    resources = load_resources()
    analysis = analyze_instruction(instruction, resources)
    print(analysis)
    evidence = collect_evidence(analysis, resources)
    types = infer_types(analysis, evidence)
    predicates = infer_predicates(analysis)
    actions = infer_actions(analysis, predicates)
    result = assemble_result(analysis, evidence, types, predicates, actions)
    save_task(result)
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
    folder_name = f"{slug}_{timestamp}"
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
