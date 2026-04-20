"""
Lightweight update checker. Compares a local version file against the
latest GitHub release. Never auto-updates — just notifies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR


VERSION_FILE = DATA_DIR / "version.json"
GITHUB_RELEASES_API = (
    "https://api.github.com/repos/SuvayanR07/"
    "vigil-adverse-event-classifier/releases/latest"
)
CHECK_TIMEOUT_S = 3


def get_local_version() -> dict:
    """Read data/version.json. Returns a default if the file is missing."""
    if not VERSION_FILE.exists():
        return {"version": "1.0.0", "meddra_terms_version": "1.0.0"}
    try:
        return json.loads(VERSION_FILE.read_text())
    except Exception:
        return {"version": "0.0.0", "meddra_terms_version": "0.0.0"}


def _parse_semver(v: str) -> tuple[int, int, int]:
    v = v.lstrip("v").strip()
    parts = v.split(".")
    try:
        return (
            int(parts[0]) if len(parts) > 0 else 0,
            int(parts[1]) if len(parts) > 1 else 0,
            int(parts[2]) if len(parts) > 2 else 0,
        )
    except ValueError:
        return (0, 0, 0)


def check_for_updates() -> dict:
    """
    Return a dict describing update status. Never raises.

    Keys:
      current_version         - local version string
      latest_version          - latest tag from GitHub, or None if unreachable
      update_available        - bool
      changelog               - release body (may be truncated)
      meddra_update_available - bool, True if the release ships newer MedDRA terms
      error                   - None or a short human-readable reason
    """
    local = get_local_version()
    current = local.get("version", "0.0.0")

    result: dict = {
        "current_version": current,
        "latest_version": None,
        "update_available": False,
        "changelog": "",
        "meddra_update_available": False,
        "error": None,
    }

    try:
        resp = requests.get(GITHUB_RELEASES_API, timeout=CHECK_TIMEOUT_S)
        if resp.status_code != 200:
            result["error"] = f"GitHub API returned HTTP {resp.status_code}"
            return result
        data = resp.json()
    except requests.exceptions.RequestException as e:
        result["error"] = f"Network error: {type(e).__name__}"
        return result
    except Exception as e:
        result["error"] = f"Unexpected error: {type(e).__name__}"
        return result

    latest = (data.get("tag_name") or "").lstrip("v")
    body = (data.get("body") or "")[:500]
    result["latest_version"] = latest
    result["changelog"] = body

    if latest and _parse_semver(latest) > _parse_semver(current):
        result["update_available"] = True
        # Heuristic: if the release body mentions MedDRA, flag it
        if "meddra" in body.lower():
            result["meddra_update_available"] = True

    return result
