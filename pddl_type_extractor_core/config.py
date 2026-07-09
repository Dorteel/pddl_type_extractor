from pathlib import Path
import yaml

DEFAULT_CONFIG = {
    "skip_arg0_role": True,
    "included_vn_thetas": [
        "Theme",
        "Patient",
        "Destination",
        "Source",
        "Location",
        "Instrument",
    ],
}


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)

    if not path.exists():
        return dict(DEFAULT_CONFIG)

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config
