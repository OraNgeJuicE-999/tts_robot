"""Named-location lookup table for VLM/voice-issued goal commands.

No ROS2 imports here on purpose -- this is pure JSON loading/lookup logic,
testable standalone, the same way dwa.py is kept separate from
dwa_controller.py.
"""

import json
from pathlib import Path
from typing import Dict, Tuple


class LocationNotFoundError(KeyError):
    """Raised when a requested location name isn't in the location map."""


def load_location_map(path) -> Dict[str, dict]:
    """Load the {name: {"x": ..., "y": ..., "yaw": ...}} JSON file at path
    into a plain dict.
    """
    text = Path(path).read_text()
    data = json.loads(text)
    return data


def get_location(location_map: Dict[str, dict], name: str) -> Tuple[float, float, float]:
    """Look up name in location_map and return (x, y, yaw)."""
    normalized = name.strip().upper()
    lookup = {k.strip().upper(): k for k in location_map}
    if normalized not in lookup:
        raise LocationNotFoundError(name)
    entry = location_map[lookup[normalized]]
    return entry["x"], entry["y"], entry.get("yaw", 0.0)