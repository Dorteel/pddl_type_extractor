#!/usr/bin/env python3
"""Interactive, ROS-free prototype for inspecting candidate planning knowledge.

Run: python remake_embedded.py
Actions and predicates from data/*.yaml are ranked using local embeddings.
Each YAML ranking includes full-instruction and individual-clause scores.
Dependencies: pip install nltk pyyaml sentence-transformers
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

LOGGER = logging.getLogger("remake_embedded")


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
# Clause splitting: sentence/semicolon boundaries plus conjunctions followed
# by a POS-tagged verb. Preserve noun lists such as 'eggs and apples'.
# This is a visible heuristic, not a dependency parser; shared arguments are
# not reconstructed. Full-instruction scoring preserves the original context.
# ---------------------------------------------------------------------------


def split_clauses(instruction: str, resources: dict) -> list[str]:
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    log_section("5. Split instruction into clauses")
    nltk = resources["nltk"]
    detokenizer = TreebankWordDetokenizer()
    clauses = []
    for sentence in nltk.sent_tokenize(instruction):
        for segment in re.split(r"[;\n]+", sentence):
            tokens = nltk.word_tokenize(segment)
            tagged = nltk.pos_tag(tokens)
            start = 0
            index = 0
            while index < len(tokens):
                token = tokens[index].lower()
                if token not in {",", "and", "then", "but", "or"}:
                    index += 1
                    continue
                next_index = index + 1
                while next_index < len(tokens) and tokens[next_index].lower() in {"and", "then", "please"}:
                    next_index += 1
                if next_index < len(tokens) and tagged[next_index][1].startswith("VB"):
                    text = detokenizer.detokenize(tokens[start:index]).strip(" ,")
                    if text:
                        clauses.append(text)
                    start = next_index
                    index = next_index
                else:
                    index += 1
            text = detokenizer.detokenize(tokens[start:]).strip(" ,")
            if text:
                clauses.append(text)
    log_data("Clauses (inspect these for splitting errors)", clauses)
    return clauses or [instruction]


# ---------------------------------------------------------------------------
# Embeddings: local Sentence Transformers inference, no API key.
# First use downloads the model; --model accepts a different model/local path.
# API reference: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install embedding dependencies: pip install sentence-transformers") from exc
    LOGGER.info("Loading embedding model %s (downloads weights if missing)", model_name)
    return SentenceTransformer(model_name)


def catalog_text(name: str, definition: dict) -> str:
    """Embed the full catalog name and intent, with explicitly declared aliases."""
    text = f"{name.replace('_', ' ')}: {definition['intent']}"
    aliases = definition.get("synonyms", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
        raise ValueError(f"Synonyms for {name} must be a list of strings")
    if aliases:
        text += " Synonyms: " + ", ".join(aliases)
    return text


# ---------------------------------------------------------------------------
# Rankings: score every entry, separately for actions and predicates.
# Overall score = maximum cosine similarity over instruction AND clauses.
# No threshold, top-k filter, domain mappings, or probability interpretation.
# Every query also gets its own full ranking. Ties sort by catalog name.
# ---------------------------------------------------------------------------


def rank_catalogs(instruction: str, clauses: list[str], actions: dict,
                  predicates: dict, model, model_name: str) -> dict:
    import numpy as np

    log_section("6. Embed and rank catalog entries")
    queries = [{"id": "instruction", "text": instruction}] + [
        {"id": f"clause_{index}", "text": text} for index, text in enumerate(clauses, 1)]
    catalogs = {"actions": actions, "predicates": predicates}
    entries = [(kind, name, catalog_text(name, definition))
               for kind, catalog in catalogs.items() for name, definition in catalog.items()]
    texts = [query["text"] for query in queries] + [text for _, _, text in entries]
    # Encode identical query texts once, while retaining their separate IDs.
    unique_texts = list(dict.fromkeys(texts))
    token_lengths = model.tokenize(unique_texts)["attention_mask"].sum(axis=1)
    for text, length in zip(unique_texts, token_lengths):
        if int(length) >= model.max_seq_length:
            LOGGER.warning("Text reaches model token limit (%s); it may be truncated: %s",
                           model.max_seq_length, text)
    vectors = np.asarray(model.encode(unique_texts, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=True))
    if vectors.ndim != 2 or len(vectors) != len(unique_texts) or not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned invalid vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embedding model returned a zero vector")
    vectors = vectors / norms
    lookup = dict(zip(unique_texts, vectors))
    query_vectors = np.stack([lookup[query["text"]] for query in queries])
    output = {}
    for kind, catalog in catalogs.items():
        names = list(catalog)
        documents = [catalog_text(name, catalog[name]) for name in names]
        scores = (np.clip(query_vectors @ np.stack([lookup[text] for text in documents]).T, -1, 1)
                  if names else np.empty((len(queries), 0)))
        overall = []
        for column, name in enumerate(names):
            best = int(np.argmax(scores[:, column]))
            overall.append({"name": name, "score": float(scores[best, column]),
                            "best_match": queries[best]["id"],
                            "instruction_score": float(scores[0, column]),
                            "clause_scores": {query["id"]: float(scores[row, column])
                                              for row, query in enumerate(queries) if row > 0}})
        overall.sort(key=lambda entry: (-entry["score"], entry["name"]))
        for rank, entry in enumerate(overall, 1):
            entry["rank"] = rank
        query_rankings = []
        for row, query in enumerate(queries):
            ordered = sorted(range(len(names)), key=lambda column: (-float(scores[row, column]), names[column]))
            query_rankings.append({**query, "ranking": [
                {"rank": rank, "name": names[column], "score": float(scores[row, column])}
                for rank, column in enumerate(ordered, 1)]})
        output[kind] = {"model": model_name, "metric": "cosine_similarity",
                        "aggregation": "maximum over full instruction and individual clauses",
                        "catalog_texts": dict(zip(names, documents)),
                        kind: overall, "query_rankings": query_rankings}
        log_data(f"{kind} overall ranking", overall)
    return output


# ---------------------------------------------------------------------------
# Pipeline: original type inference, followed by independent semantic rankings.
# ---------------------------------------------------------------------------


def process_instruction(instruction: str, data_dir: str | Path | None = None,
                        tasks_dir: str | Path | None = None,
                        model_name: str = DEFAULT_MODEL, model=None) -> dict:
    if not instruction.strip():
        raise ValueError("Instruction must not be empty.")
    action_catalog, predicate_catalog = load_catalogs(data_dir)
    if model is None:
        model = load_embedding_model(model_name)
    resources = load_resources()
    analysis = analyze_instruction(instruction, resources)
    evidence = collect_evidence(analysis, resources)
    types = infer_types(analysis, evidence)
    clauses = split_clauses(instruction, resources)
    rankings = rank_catalogs(instruction, clauses, action_catalog, predicate_catalog, model, model_name)
    result = {"instruction": instruction, "types": types, **rankings}
    save_task(result, tasks_dir)
    return result


# ---------------------------------------------------------------------------
# Task files: one unique instruction folder, with readable YAML candidates.
# Timestamps track runs; numeric suffixes handle runs within the same second.
# ---------------------------------------------------------------------------


def save_task(result: dict, tasks_dir: str | Path | None = None) -> Path:
    """Save the instruction, ranked matches and type declarations."""
    import yaml

    log_section("8. Save task files")
    root = Path(tasks_dir) if tasks_dir is not None else Path(__file__).resolve().parent / "tasks"
    root.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result["instruction"].lower()).strip("-")[:80].rstrip("-") or "instruction"
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S%z")
    folder_name = f"{slug}_remake_embedded_{timestamp}"
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
        payload = ({"types": list(dict.fromkeys(entry["declaration"] for entry in result["types"]))}
                   if name == "types" else result[name])
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        LOGGER.debug("Wrote %s", path)
    LOGGER.info("Task saved to %s", folder)
    return folder


# ---------------------------------------------------------------------------
# Standalone interactive entry point: ask first, then execute the pipeline.
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers model name or local path")
    args = parser.parse_args()
    configure_logging()
    try:
        instruction = input("Enter an instruction: ").strip()
        result = process_instruction(instruction, model_name=args.model)
        log_section("9. Final candidate model (full evidence appears above)")
        print(json.dumps({key: result[key] for key in
                          ("instruction", "types", "predicates", "actions")}, indent=2))
    except (EOFError, KeyboardInterrupt):
        print("\nStopped.")
    except Exception:
        LOGGER.exception("Extraction failed. See the traceback for the failing stage.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
