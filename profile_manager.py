"""
profile_manager.py — JSON Profile System
Each profile is a JSON file mapping gesture names → action names.
Profiles are stored in ../profiles/<name>.json

Python Data Structures used:
  • dict  — gesture → action mapping (O(1) lookup)
  • list  — profile name enumeration
  • str   — JSON serialisation via json module
"""

import os
import json

PROFILES_DIR = "../profiles"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _profile_path(name: str) -> str:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return os.path.join(PROFILES_DIR, f"{name}.json")


def load_profile(name: str) -> dict:
    """Load profile by name. Returns empty dict if not found."""
    path = _profile_path(name)
    if not os.path.exists(path):
        # Auto-create default profile on first run
        if name == "default":
            default = {
                "thumbs_up":   "scroll_up",
                "thumbs_down": "scroll_down",
                "open_palm":   "screenshot",
                "fist":        "play_pause",
                "peace":       "volume_up",
            }
            save_profile("default", default)
            return default
        return {}
    with open(path) as f:
        return json.load(f)


def save_profile(name: str, mapping: dict):
    """Persist a profile dict to disk."""
    with open(_profile_path(name), "w") as f:
        json.dump(mapping, f, indent=2)


def list_profiles() -> list:
    """Return names of all saved profiles (without .json extension)."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return [
        fn.replace(".json", "")
        for fn in os.listdir(PROFILES_DIR)
        if fn.endswith(".json")
    ]


def delete_profile(name: str) -> bool:
    path = _profile_path(name)
    if os.path.exists(path) and name != "default":
        os.remove(path)
        return True
    return False


def export_profile(name: str) -> str:
    """Return a JSON string of the profile (for download)."""
    return json.dumps(load_profile(name), indent=2)


def import_profile(name: str, json_str: str):
    """Import a profile from a JSON string."""
    mapping = json.loads(json_str)
    save_profile(name, mapping)
