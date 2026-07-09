from nltk.corpus import propbank
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import nltk

for pkg in ["propbank", "wordnet", "punkt", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()


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

    return output


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
