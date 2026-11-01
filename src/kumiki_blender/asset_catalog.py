"""Browsable, filesystem-backed index of reusable 3D asset *families*.

The catalog stores one manifest per family. Parameterized builds may emit many
asset IDs, but their family has one stable searchable location. Geometry and
Blender adapters remain separate from this discovery layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


CATALOG_ROOT = Path(__file__).resolve().parent / "assets"
_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_REVISION = re.compile(r"r[0-9]{2,6}\Z")
_PLAN_COMMANDS = {"validate", "plan-portico", "plan-facade", "plan-monument",
                  "plan-biplane", "plan-classical-column"}
_BUILD_COMMANDS = {"build", "build-portico", "build-facade",
                   "build-monument", "build-biplane"}


class CatalogError(ValueError):
    """An asset manifest or requested catalog operation is invalid."""


@dataclass(frozen=True)
class AssetEntry:
    asset_id: str
    title: str
    category: str
    subcategory: str
    description: str
    kind: str
    tags: tuple[str, ...]
    aliases: tuple[str, ...]
    rigged: bool
    animated: bool
    sources: tuple[str, ...]
    plan_command: tuple[str, ...] | None
    build_command: tuple[str, ...] | None
    manifest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.asset_id, "title": self.title,
            "category": self.category, "subcategory": self.subcategory,
            "description": self.description, "kind": self.kind,
            "tags": list(self.tags), "aliases": list(self.aliases),
            "rigged": self.rigged, "animated": self.animated,
            "sources": list(self.sources),
            "plan": list(self.plan_command) if self.plan_command else None,
            "build": list(self.build_command) if self.build_command else None,
            "manifest": self.manifest,
        }


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value):
        raise CatalogError(f"{label} must be a lowercase hyphenated slug")
    return value


def _strings(value: Any, label: str, *, slugs: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip()
                                               for item in value):
        raise CatalogError(f"{label} must be a list of non-empty strings")
    result = tuple(_slug(item, label) if slugs else item.strip() for item in value)
    if len(result) != len(set(result)):
        raise CatalogError(f"{label} contains duplicates")
    return result


def _source_path(value: str) -> str:
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or any(part in (".", "..", "")
                                                   for part in value.split("/"))
            or "\\" in value or not path.suffix):
        raise CatalogError(f"Unsafe source path: {value!r}")
    return value


def _command(value: Any, label: str, allowed: set[str]) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = _strings(value, label)
    if not parts or parts[0] not in allowed or "--output" in parts:
        raise CatalogError(f"{label} must start with a supported CLI command and omit --output")
    return parts


def _load_entry(path: Path, root: Path) -> AssetEntry:
    relative = path.relative_to(root)
    if len(relative.parts) != 4 or relative.name != "asset.json":
        raise CatalogError(f"Unexpected catalog location: {relative}")
    category, subcategory, directory, _ = relative.parts
    for label, value in (("category", category), ("subcategory", subcategory),
                         ("asset directory", directory)):
        _slug(value, label)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CatalogError(f"Cannot read {relative}: {error}") from error
    expected = {"schema_version", "id", "title", "description", "kind", "tags",
                "aliases", "rigged", "animated", "sources", "plan", "build"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise CatalogError(f"{relative}: expected exact fields {', '.join(sorted(expected))}")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise CatalogError(f"{relative}: unsupported schema_version")
    asset_id = _slug(raw["id"], "id")
    if asset_id != directory:
        raise CatalogError(f"{relative}: id must match its directory")
    if raw["kind"] not in ("procedural", "scanned", "imported", "hybrid"):
        raise CatalogError(f"{relative}: unknown asset kind")
    for field in ("title", "description"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise CatalogError(f"{relative}: {field} must not be empty")
    if type(raw["rigged"]) is not bool or type(raw["animated"]) is not bool:
        raise CatalogError(f"{relative}: rigged and animated must be booleans")
    tags = _strings(raw["tags"], "tags", slugs=True)
    if not tags:
        raise CatalogError(f"{relative}: at least one tag is required")
    sources = tuple(_source_path(value) for value in _strings(raw["sources"], "sources"))
    if not sources:
        raise CatalogError(f"{relative}: at least one source is required")
    return AssetEntry(
        asset_id, raw["title"].strip(), category, subcategory,
        raw["description"].strip(), raw["kind"], tags,
        _strings(raw["aliases"], "aliases"), raw["rigged"], raw["animated"],
        sources, _command(raw["plan"], "plan", _PLAN_COMMANDS),
        _command(raw["build"], "build", _BUILD_COMMANDS),
        f"src/kumiki_blender/assets/{relative.as_posix()}",
    )


def load_catalog(root: Path | None = None) -> tuple[AssetEntry, ...]:
    """Load every manifest without a hand-maintained central index."""
    root = (root or CATALOG_ROOT).resolve()
    if not root.is_dir():
        raise CatalogError(f"Asset catalog not found: {root}")
    entries: list[AssetEntry] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("asset.json")):
        if not path.resolve().is_relative_to(root):
            raise CatalogError(f"Catalog symlink escapes root: {path}")
        entry = _load_entry(path, root)
        if entry.asset_id in seen:
            raise CatalogError(f"Duplicate catalog ID: {entry.asset_id}")
        seen.add(entry.asset_id)
        entries.append(entry)
    if not entries:
        raise CatalogError(f"Asset catalog is empty: {root}")
    return tuple(sorted(entries, key=lambda entry: (entry.category,
                                                     entry.subcategory, entry.asset_id)))


def find_asset(entries: tuple[AssetEntry, ...], asset_id: str) -> AssetEntry:
    for entry in entries:
        if entry.asset_id == asset_id:
            return entry
    raise CatalogError(f"Unknown asset: {asset_id}. Use 'catalog search <words>' to find it")


def search_assets(entries: tuple[AssetEntry, ...], query: str = "", *,
                  category: str | None = None, tag: str | None = None,
                  rigged: bool = False) -> tuple[AssetEntry, ...]:
    """Case-insensitive multiword search across titles, IDs, aliases and tags."""
    words = re.findall(r"[a-z0-9]+", query.casefold())
    matches = []
    for entry in entries:
        if category and entry.category != category:
            continue
        if tag and tag not in entry.tags:
            continue
        if rigged and not entry.rigged:
            continue
        haystack = " ".join((entry.asset_id, entry.title, entry.description,
                             entry.category, entry.subcategory, *entry.tags, *entry.aliases)).casefold()
        if all(word in haystack for word in words):
            matches.append(entry)
    return tuple(matches)


def revision_path(entry: AssetEntry, revision: str, root: Path = Path(".review")) -> Path:
    if not _REVISION.fullmatch(revision):
        raise CatalogError("revision must be r followed by 2–6 digits, e.g. r01 or r002")
    return root / entry.category / entry.subcategory / entry.asset_id / revision


def validate_sources(entries: tuple[AssetEntry, ...], repository_root: Path) -> None:
    """Verify that checked-in manifests point to real, in-repository sources."""
    repository_root = repository_root.resolve()
    for entry in entries:
        for value in entry.sources:
            target = (repository_root / value).resolve()
            if not target.is_relative_to(repository_root) or not target.is_file():
                raise CatalogError(f"{entry.asset_id}: missing/escaping source {value}")
