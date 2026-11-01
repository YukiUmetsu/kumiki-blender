"""Dependency-free interchange between validated authoring and Blender workers.

The host can use Pydantic to validate and compile a recipe. Blender consumes
this bounded data-only plan without importing Pydantic or any other pip wheel.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .architecture_geometry import (Mesh, MeshInstance, audit_mesh, place_mesh,
                                    world_vertices)
from .composer import AssemblyPlan
from .surfaces import get_surface

if TYPE_CHECKING:
    from .asset_spec import AssetSpec


_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PRESETS = {"painted-steel", "galvanized-steel", "cast-concrete", "natural-stone",
            "weathered-stone", "timber", "optical-glass"}
_KINDS = {"solid.box", "solid.lathe", "solid.hollow-lathe", "mechanical.rod", "detail.radial-studs"}


class CompiledPlanError(ValueError):
    """A compiled asset plan is invalid or was altered after preparation."""


@dataclass(frozen=True)
class MaterialRecord:
    preset: str
    color: str | None
    roughness: float | None


@dataclass(frozen=True)
class QualityRecord:
    preset: str
    views: tuple[str, ...]


@dataclass(frozen=True)
class ComponentRecord:
    id: str
    type: str


@dataclass(frozen=True)
class CompiledSpec:
    asset_id: str
    materials: dict[str, MaterialRecord]
    quality: QualityRecord
    components: tuple[ComponentRecord, ...]


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def write_compiled_plan(spec: "AssetSpec", plan: AssemblyPlan, output: Path) -> None:
    """Materialize a portable mesh plan without replacing an existing file."""
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise CompiledPlanError(f"Compiled plan already exists: {path}")
    meshes: list[dict] = []
    indexes: dict[int, int] = {}
    instances = []
    for part in plan.instances:
        identity = id(part.mesh)
        if identity not in indexes:
            indexes[identity] = len(meshes)
            meshes.append({"vertices": part.mesh.vertices, "faces": part.mesh.faces})
        component_id = next((item.id for item in spec.components if part.name == item.id or
                             (item.type == "detail.radial-studs" and
                              part.name.startswith(item.id + "-"))), None)
        if component_id is None:
            raise CompiledPlanError(f"Missing origin component for {part.name}")
        instances.append({
            "name": part.name, "mesh": indexes[identity], "component_id": component_id,
            "material": part.material, "origin": part.origin,
            "axis_x": part.axis_x, "axis_y": part.axis_y, "axis_z": part.axis_z,
        })
    data = {
        "compiled_version": 1,
        "spec": {
            "asset_id": spec.asset_id,
            "materials": {role: item.model_dump(mode="json") for role, item in spec.materials.items()},
            "quality": spec.quality.model_dump(mode="json"),
            "components": [{"id": item.id, "type": item.type} for item in spec.components],
        },
        "plan": {
            "meshes": meshes, "instances": instances,
            "bounds_min": plan.bounds_min, "bounds_max": plan.bounds_max,
            "named_parts": plan.named_parts,
            "recipe_sha256": plan.recipe_sha256,
        },
    }
    data["sha256"] = hashlib.sha256(_canonical(data)).hexdigest()
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def _exact(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise CompiledPlanError(f"{label}: invalid object fields")
    return value


def _slug(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or not _SLUG.fullmatch(value):
        raise CompiledPlanError(f"{label}: invalid identifier")
    return value


def _vector(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise CompiledPlanError(f"{label}: expected {size} coordinates")
    if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 100_000 for x in value):
        raise CompiledPlanError(f"{label}: invalid or nonfinite coordinate")
    return tuple(float(x) for x in value)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    data = {}
    for key, value in pairs:
        if key in data:
            raise CompiledPlanError(f"Duplicate compiled JSON property: {key}")
        data[key] = value
    return data


def _load_spec(raw: dict) -> CompiledSpec:
    data = _exact(raw, {"asset_id", "materials", "quality", "components"}, "compiled spec")
    asset_id = _slug(data["asset_id"], "asset_id")
    raw_materials = data["materials"]
    if not isinstance(raw_materials, dict) or not 1 <= len(raw_materials) <= 32:
        raise CompiledPlanError("materials: expected 1–32 declared material roles")
    materials = {}
    for role, raw_material in raw_materials.items():
        _slug(role, "material role")
        material = _exact(raw_material, {"preset", "color", "roughness"}, "material")
        if material["preset"] not in _PRESETS:
            raise CompiledPlanError(f"{role}: unsupported material preset")
        get_surface(material["preset"], color=material["color"], roughness=material["roughness"])
        materials[role] = MaterialRecord(**material)
    raw_quality = _exact(data["quality"], {"preset", "views"}, "quality")
    if raw_quality["preset"] not in ("smoke", "production", "cinematic-hero"):
        raise CompiledPlanError("quality: unsupported preset")
    views = raw_quality["views"]
    if (not isinstance(views, list) or not 1 <= len(views) <= 4 or
            views[0] != "hero" or len(views) != len(set(views)) or
            any(v not in ("hero", "front", "side", "detail") for v in views)):
        raise CompiledPlanError("quality: invalid or duplicated inspection views")
    raw_components = data["components"]
    if not isinstance(raw_components, list) or not 1 <= len(raw_components) <= 256:
        raise CompiledPlanError("components: expected 1–256 component declarations")
    components = []
    for raw_component in raw_components:
        component = _exact(raw_component, {"id", "type"}, "component")
        cid = _slug(component["id"], "component id")
        if component["type"] not in _KINDS:
            raise CompiledPlanError(f"{cid}: unsupported component type")
        components.append(ComponentRecord(cid, component["type"]))
    if len({component.id for component in components}) != len(components):
        raise CompiledPlanError("component ids must be unique")
    return CompiledSpec(asset_id, materials, QualityRecord(raw_quality["preset"], tuple(views)),
                        tuple(components))


def load_compiled_plan(source: str | Path) -> tuple[CompiledSpec, AssemblyPlan]:
    """Reject corrupted or hazardous interchange files before Blender writes."""
    path = Path(source)
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise CompiledPlanError("Compiled plan exceeds the 64 MiB safety limit")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(CompiledPlanError(
                             f"Nonfinite compiled number: {value}")))
        root = _exact(raw, {"compiled_version", "spec", "plan", "sha256"}, "compiled plan")
        if type(root["compiled_version"]) is not int or root["compiled_version"] != 1:
            raise CompiledPlanError("Unsupported compiled plan version")
        checksum = root["sha256"]
        if not isinstance(checksum, str) or not _DIGEST.fullmatch(checksum):
            raise CompiledPlanError("Compiled plan digest missing")
        if hashlib.sha256(_canonical({k: v for k, v in root.items() if k != "sha256"})).hexdigest() != checksum:
            raise CompiledPlanError("Compiled plan digest mismatch")
        spec = _load_spec(root["spec"])
        data = _exact(root["plan"], {"meshes", "instances", "bounds_min", "bounds_max",
                                     "named_parts", "recipe_sha256"}, "scene plan")
        if not isinstance(data["meshes"], list) or not 1 <= len(data["meshes"]) <= 4096:
            raise CompiledPlanError("Expected 1–4096 mesh definitions")
        meshes = []
        for raw_mesh in data["meshes"]:
            item = _exact(raw_mesh, {"vertices", "faces"}, "mesh")
            if not isinstance(item["vertices"], list) or not 4 <= len(item["vertices"]) <= 200_000:
                raise CompiledPlanError("Mesh vertex count outside safe bounds")
            if not isinstance(item["faces"], list) or not 4 <= len(item["faces"]) <= 400_000:
                raise CompiledPlanError("Mesh face count outside safe bounds")
            verts = tuple(_vector(v, 3, "mesh vertex") for v in item["vertices"])
            faces = []
            for face in item["faces"]:
                if (not isinstance(face, list) or not 3 <= len(face) <= 256 or
                        any(type(i) is not int or i < 0 or i >= len(verts) for i in face)):
                    raise CompiledPlanError("Invalid face or vertex index")
                faces.append(tuple(face))
            mesh = Mesh(verts, tuple(faces))
            report = audit_mesh(mesh)
            if report.boundary_edges or report.nonmanifold_edges or not report.volume_m3 > 0:
                raise CompiledPlanError("Invalid mesh topology or signed volume")
            meshes.append(mesh)
        raw_instances = data["instances"]
        if not isinstance(raw_instances, list) or not 1 <= len(raw_instances) <= 4096:
            raise CompiledPlanError("Expected 1–4096 placed instances")
        by_id = {component.id: component for component in spec.components}
        instances: list[MeshInstance] = []
        for raw_instance in raw_instances:
            item = _exact(raw_instance, {"name", "mesh", "component_id", "material", "origin",
                                         "axis_x", "axis_y", "axis_z"}, "instance")
            cid = _slug(item["component_id"], "component id")
            component = by_id.get(cid)
            if component is None:
                raise CompiledPlanError("Instance references missing component")
            if item["name"] != cid and not (component.type == "detail.radial-studs" and
                                               isinstance(item["name"], str) and
                                               re.fullmatch(re.escape(cid)+r"-[0-9]{3}", item["name"])):
                raise CompiledPlanError("Instance name does not match its component")
            if item["material"] not in spec.materials:
                raise CompiledPlanError("Instance material role is missing")
            index = item["mesh"]
            if type(index) is not int or not 0 <= index < len(meshes):
                raise CompiledPlanError("Invalid mesh index")
            instances.append(place_mesh(
                meshes[index], name=item["name"], material=item["material"],
                origin=_vector(item["origin"], 3, "origin"),
                axis_x=_vector(item["axis_x"], 3, "axis X"),
                axis_y=_vector(item["axis_y"], 3, "axis Y"),
                axis_z=_vector(item["axis_z"], 3, "axis Z"),
            ))
        if len({item.name for item in instances}) != len(instances):
            raise CompiledPlanError("Repeated instance names")
        lo = _vector(data["bounds_min"], 3, "bounds_min")
        hi = _vector(data["bounds_max"], 3, "bounds_max")
        actual_lo = [float("inf")] * 3
        actual_hi = [float("-inf")] * 3
        for instance in instances:
            for vertex in world_vertices(instance):
                for axis in range(3):
                    actual_lo[axis] = min(actual_lo[axis], vertex[axis])
                    actual_hi[axis] = max(actual_hi[axis], vertex[axis])
        if any(abs(actual_lo[a]-lo[a]) > 1e-5 or abs(actual_hi[a]-hi[a]) > 1e-5 for a in range(3)):
            raise CompiledPlanError("Scene bounds do not match generated geometry")
        named = data["named_parts"]
        if (not isinstance(named, list) or any(not isinstance(item, list) or len(item) != 2 or
                item[0] not in _KINDS or type(item[1]) is not int or item[1] < 1 for item in named) or
                sum(item[1] for item in named) != len(instances)):
            raise CompiledPlanError("Invalid component instance counts")
        recipe_digest = data["recipe_sha256"]
        if not isinstance(recipe_digest, str) or not _DIGEST.fullmatch(recipe_digest):
            raise CompiledPlanError("Invalid recipe digest")
        plan = AssemblyPlan(spec.asset_id, tuple(instances), lo, hi,
                            tuple((item[0], item[1]) for item in named), recipe_digest)
        return spec, plan
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, CompiledPlanError):
            raise
        raise CompiledPlanError(f"Invalid compiled plan {path}: {error}") from error
