"""Local-only, private defaults for repeated statue reconstruction runs.

The registry is deliberately scoped to the current working project's ignored
`.review/` directory. It never scans a user's home for unrelated photographs.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .references import ReferenceError


_SAFE_ASSET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _asset_dir(asset_id: str, workspace: Path) -> Path:
    if not isinstance(asset_id, str) or not _SAFE_ASSET.fullmatch(asset_id):
        raise ReferenceError("Asset ID must use 1–64 letters, numbers, underscores or hyphens")
    review = workspace / ".review"
    target = review / asset_id
    if review.is_symlink() or target.is_symlink():
        raise ReferenceError("Asset registry path cannot contain symbolic links")
    return target


def _read_previous(root: Path, asset_id: str) -> dict | None:
    path = root / "asset-state.json"
    if path.is_symlink():
        raise ReferenceError("Asset state must not be a symbolic link")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        raise ReferenceError(f"Invalid saved asset state: {path}") from error
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("asset_id") != asset_id
            or not isinstance(data.get("photos_dir"), str)):
        raise ReferenceError(f"Saved asset state does not match {asset_id}")
    return data


def resolve_statue_inputs(asset_id: str, photos: Path | None, output: Path | None,
                          *, workspace: Path | None = None,
                          resume: bool = False) -> tuple[Path, Path]:
    """Reuse an asset's source, choosing a fresh output unless resuming."""
    cwd = (workspace or Path.cwd()).resolve()
    asset_root = _asset_dir(asset_id, cwd)
    saved = _read_previous(asset_root, asset_id)
    if photos is None:
        if saved is None:
            raise ReferenceError(f"No saved photo folder for {asset_id}; supply --photos once")
        photos = Path(saved["photos_dir"])
    if photos.is_symlink() or not photos.is_dir():
        raise ReferenceError(f"Photo directory does not exist or is unsafe: {photos}")
    photos = photos.resolve()
    if output is None:
        if resume:
            if not saved or not isinstance(saved.get("last_output"), str):
                raise ReferenceError("No saved output revision to resume; supply --output")
            output = Path(saved["last_output"])
        else:
            for index in range(1, 10001):
                candidate = asset_root / f"statue-hq-r{index:03d}"
                if not candidate.exists() and not candidate.is_symlink():
                    output = candidate
                    break
            else:
                raise ReferenceError("All automatic revision numbers are in use; supply --output")
    if output.is_symlink() or output.resolve().is_relative_to(photos):
        raise ReferenceError("Output must not be a symlink or inside the original photo folder")
    return photos, output


def remember_statue_inputs(asset_id: str, photos: Path, output: Path,
                           *, workspace: Path | None = None) -> Path:
    """Atomically store validated local pointers, including failed run attempts."""
    cwd = (workspace or Path.cwd()).resolve()
    root = _asset_dir(asset_id, cwd)
    if root.exists() and not root.is_dir():
        raise ReferenceError("Asset registry location is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    state = root / "asset-state.json"
    if state.is_symlink():
        raise ReferenceError("Asset state cannot be a symbolic link")
    data = {"schema_version": 1, "asset_id": asset_id,
            "photos_dir": str(photos.resolve()), "last_output": str(output.resolve())}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".asset-", suffix=".json",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, state)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return state
