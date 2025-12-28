import json
import os
import hashlib
from typing import Dict, Any, Optional
from utils.logger import logger

SCHEMA_VERSION = 1


def compute_file_sha1(file_path: str) -> str:
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def build_file_record(file_path: str) -> Dict[str, Any]:
    return {
        "mtime": os.path.getmtime(file_path),
        "size": os.path.getsize(file_path),
        "sha1": compute_file_sha1(file_path),
    }


def load_registry(registry_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(registry_path):
        return None
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load registry {registry_path}: {e}")
        return None


def save_registry(registry_path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
