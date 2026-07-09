from .config import load_config
from pddl_type_extractor_core.extractor import instruction_to_pb_vn_mappings, instruction_to_propbank_debug
from pddl_type_extractor_core.type_reasoner import derive_types_from_pb_vn_mappings

def derive_types_for_instruction(instruction: str, config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    mappings = instruction_to_pb_vn_mappings(instruction)
    types = derive_types_from_pb_vn_mappings(mappings, config)
    pb_frames, pb_role_descriptions, vn_classes = instruction_to_propbank_debug(instruction)

    return {
        "instruction": instruction,
        "types": types,
        "propbank_frames_considered": pb_frames,
        "propbank_role_descriptions_considered": pb_role_descriptions,
        "verbnet_classes_considered": vn_classes,
    }
