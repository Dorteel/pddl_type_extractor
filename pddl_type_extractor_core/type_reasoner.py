import re
import nltk
from nltk.corpus import verbnet as vn
from nltk.stem import WordNetLemmatizer

for pkg in ["verbnet", "wordnet"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()

STANDARD_TYPES = {
    "object",
    "locatable - object",
    "location - object",
    "robot - locatable",
    "item - locatable",
}

PREDICATE_TO_AFFORDANCE = {
    "motion": "movable",
}

CONTAINER_PREPOSITIONS = {"in", "into", "inside", "within"}

def infer_container_type(role):
    prep = role.get("preposition")
    vn_thetas = {vn_role.get("vn_theta") for vn_role in role.get("vn_roles", [])}

    if prep in CONTAINER_PREPOSITIONS and vn_thetas & {"Destination", "Location", "Goal"}:
        return "container - item"

    return None

def should_skip_role(role: dict, config: dict) -> bool:
    if config.get("skip_arg0_role", True) and role.get("arg") == "0":
        return True

    included = set(config.get("included_vn_thetas", []))
    vn_thetas = {vn_role["vn_theta"] for vn_role in role.get("vn_roles", [])}

    if vn_thetas and not (vn_thetas & included):
        return True

    return False


def verb_to_able_type(verb: str) -> str:
    lemma = lemmatizer.lemmatize(verb.lower(), "v")

    if lemma.endswith("e"):
        return lemma[:-1] + "able"

    return lemma + "able"


def affordance_from_role_description(descr: str) -> str | None:
    if not descr:
        return None

    descr = descr.lower().strip()

    patterns = [
        r"thing to ([a-z]+)",
        r"thing being ([a-z]+ed)",
        r"thing ([a-z]+ed)",
        r"thing ([a-z]+ing)",
        r"thing ([a-z]+)",
        r"entity to ([a-z]+)",
        r"entity being ([a-z]+ed)",
        r"entity ([a-z]+ed)",
        r"entity ([a-z]+ing)",
        r"entity ([a-z]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, descr)
        if match:
            verb_form = match.group(1)
            verb_lemma = lemmatizer.lemmatize(verb_form, "v")
            return verb_to_able_type(verb_lemma)

    if re.fullmatch(r"[a-z]+ed", descr):
        verb_lemma = lemmatizer.lemmatize(descr, "v")
        return verb_to_able_type(verb_lemma)

    if re.fullmatch(r"[a-z]+", descr):
        return descr

    return None


def predicate_mentions_themrole(predicate: dict, themrole: str) -> bool:
    for arg in predicate.get("arguments", []):
        if arg.get("type") == "ThemRole" and arg.get("value", "").strip() == themrole:
            return True

    return False


def vn_role_to_affordance_types(vn_class: str, vn_theta: str) -> set[str]:
    derived = set()

    try:
        frames = vn.frames(vn_class)
    except Exception:
        return derived

    for frame in frames:
        for predicate in frame.get("semantics", []):
            pred_name = predicate.get("predicate_value")

            if pred_name in PREDICATE_TO_AFFORDANCE:
                if predicate_mentions_themrole(predicate, vn_theta):
                    derived.add(PREDICATE_TO_AFFORDANCE[pred_name])

    return derived


def as_item_type(type_name: str) -> str:
    if " - " in type_name:
        return type_name

    if type_name in {"object", "locatable", "location", "robot", "item"}:
        return type_name

    return f"{type_name} - item"


def derive_types_from_pb_vn_mappings(mappings: dict, config: dict) -> list[str]:
    types = set(STANDARD_TYPES)

    for verb, rolesets in mappings.items():
        for roleset in rolesets:
            for role in roleset["roles"]:
                if should_skip_role(role, config):
                    continue

                container_type = infer_container_type(role)
                if container_type:
                    types.add(container_type)

                vn_affordances = set()

                for vn_role in role["vn_roles"]:
                    vn_class = vn_role["vn_class"]
                    vn_theta = vn_role["vn_theta"]

                    vn_affordances |= vn_role_to_affordance_types(vn_class, vn_theta)

                for affordance_type in vn_affordances:
                    types.add(as_item_type(affordance_type))

                # Only use PropBank descriptions if VerbNet did not already give
                # a stronger semantic affordance.
                if "movable" not in vn_affordances:
                    role_type = affordance_from_role_description(role["descr"])
                    if role_type:
                        types.add(as_item_type(role_type))

    return sorted(types)