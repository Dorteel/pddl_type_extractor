from nltk.corpus import propbank
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import nltk
import re

for pkg in ["propbank", "wordnet", "punkt", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()

CONTAINER_PREPOSITIONS = {"in", "into", "inside", "within"}
DESTINATION_THETAS = {"Destination", "Location", "Goal"}


def extract_prepositions(instruction: str) -> set[str]:
    tokens = re.findall(r"\b[a-zA-Z]+\b", instruction.lower())
    return {tok for tok in tokens if tok in CONTAINER_PREPOSITIONS}


def role_has_vn_theta(role: dict, target_thetas: set[str]) -> bool:
    return any(
        vn_role.get("vn_theta") in target_thetas
        for vn_role in role.get("vn_roles", [])
    )


def add_instruction_prepositions(mappings: dict, instruction: str) -> dict:
    preps = extract_prepositions(instruction)

    container_preps = preps & CONTAINER_PREPOSITIONS
    if not container_preps:
        return mappings

    selected_prep = sorted(container_preps)[0]

    for verb, rolesets in mappings.items():
        for roleset in rolesets:
            for role in roleset.get("roles", []):
                if role_has_vn_theta(role, DESTINATION_THETAS):
                    role["preposition"] = selected_prep

    return mappings


def extract_verbs(instruction: str) -> list[str]:
    tokens = word_tokenize(instruction)
    tagged = pos_tag(tokens)
    verbs = [lemmatizer.lemmatize(tokens[0].lower(), "v")] if tokens else []

    for token, tag in tagged:
        if tag.startswith("VB"):
            lemma = lemmatizer.lemmatize(token.lower(), "v")
            if lemma not in verbs:
                verbs.append(lemma)

    return verbs


def get_vn_roles(role) -> list[dict]:
    return [
        {
            "vn_class": vnrole.attrib.get("vncls"),
            "vn_theta": vnrole.attrib.get("vntheta"),
        }
        for vnrole in role.findall("vnrole")
        if vnrole.attrib.get("vncls") and vnrole.attrib.get("vntheta")
    ]


def roleset_to_pb_vn_mapping(roleset) -> dict | None:
    roles = []

    for role in roleset.findall("roles/role"):
        vn_roles = get_vn_roles(role)
        if not vn_roles:
            continue

        roles.append(
            {
                "arg": role.attrib.get("n"),
                "descr": role.attrib.get("descr"),
                "function": role.attrib.get("f"),
                "vn_roles": vn_roles,
            }
        )

    if not roles:
        return None

    return {
        "roleset": roleset.attrib["id"],
        "name": roleset.attrib.get("name"),
        "vn_classes": sorted(
            {
                vn_role["vn_class"]
                for role in roles
                for vn_role in role["vn_roles"]
            }
        ),
        "roles": roles,
    }


def instruction_to_pb_vn_mappings(instruction: str) -> dict:
    output = {}

    for verb in extract_verbs(instruction):
        mapped_rolesets = []

        for roleset in propbank.rolesets():
            roleset_id = roleset.attrib["id"]

            if roleset_id.startswith(verb + "."):
                mapped = roleset_to_pb_vn_mapping(roleset)
                if mapped is not None:
                    mapped_rolesets.append(mapped)

        if mapped_rolesets:
            output[verb] = mapped_rolesets

    mappings = add_instruction_prepositions(output, instruction)
    return mappings


def instruction_to_propbank_debug(instruction: str) -> tuple[list, list, list]:
    propbank_frames = []
    propbank_role_descriptions = []
    verbnet_classes = set()

    for verb in extract_verbs(instruction):
        for roleset in propbank.rolesets():
            roleset_id = roleset.attrib["id"]

            if roleset_id.startswith(verb + "."):
                propbank_frames.append(roleset_id)

                roles = []
                for role in roleset.findall("roles/role"):
                    roles.append(
                        {
                            "arg": role.attrib.get("n"),
                            "descr": role.attrib.get("descr"),
                            "function": role.attrib.get("f"),
                        }
                    )

                propbank_role_descriptions.append(
                    {
                        "roleset": roleset_id,
                        "name": roleset.attrib.get("name"),
                        "roles": roles,
                    }
                )

                mapped = roleset_to_pb_vn_mapping(roleset)
                if mapped is not None:
                    verbnet_classes.update(mapped["vn_classes"])

    return (
        sorted(set(propbank_frames)),
        propbank_role_descriptions,
        sorted(verbnet_classes),
    )
