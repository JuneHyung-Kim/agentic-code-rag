import json
import os
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from utils.logger import logger

SCHEMA_VERSION = 2


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


def _migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate schema version 1 to version 2."""
    root_path = data.get("root_path")
    files = data.get("files", {})

    if not root_path:
        logger.warning("Cannot migrate v1 registry: missing root_path")
        return _create_empty_registry()

    logger.info(f"Migrating registry from v1 to v2 for project: {root_path}")
    return {
        "schema_version": 2,
        "projects": {
            root_path: {
                "indexed_at": datetime.now().isoformat(),
                "files": files
            }
        }
    }


def _create_empty_registry() -> Dict[str, Any]:
    """Create a new empty registry with v2 schema."""
    return {
        "schema_version": 2,
        "projects": {}
    }


def load_registry(registry_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(registry_path):
        return None
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        # Handle schema migration
        schema_version = data.get("schema_version", 1)

        if schema_version == 1:
            # Auto-migrate v1 to v2
            data = _migrate_v1_to_v2(data)
            logger.info("Registry migrated from v1 to v2")
        elif schema_version == 2:
            # Already v2, ensure structure is valid
            if "projects" not in data:
                logger.warning("Invalid v2 registry structure, creating new")
                return _create_empty_registry()
        else:
            logger.warning(f"Unknown schema version {schema_version}, creating new registry")
            return _create_empty_registry()

        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load registry {registry_path}: {e}")
        return None


def save_registry(registry_path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_project_files(registry: Optional[Dict[str, Any]], project_root: str) -> Dict[str, Any]:
    """Get files for a specific project from registry."""
    if not registry or registry.get("schema_version") != 2:
        return {}

    projects = registry.get("projects", {})
    project_data = projects.get(project_root, {})
    return project_data.get("files", {})


def update_project_files(registry: Dict[str, Any], project_root: str, files: Dict[str, Any]) -> None:
    """Update files for a specific project in registry."""
    if registry.get("schema_version") != 2:
        raise ValueError("Registry must be schema version 2")

    if "projects" not in registry:
        registry["projects"] = {}

    registry["projects"][project_root] = {
        "indexed_at": datetime.now().isoformat(),
        "files": files
    }


def remove_project(registry: Dict[str, Any], project_root: str) -> None:
    """Remove a project from registry."""
    if registry.get("schema_version") != 2:
        return

    projects = registry.get("projects", {})
    if project_root in projects:
        del projects[project_root]
        logger.info(f"Removed project from registry: {project_root}")
