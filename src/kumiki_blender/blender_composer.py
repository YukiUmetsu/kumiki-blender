"""Production Blender adapter for declarative, reusable component assemblies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .blender_output import aim_at, create_model_scene, export_revision
from .blender_material_packs import create_texture_pack_material
from .blender_scene import _create_box, _material
from .blender_textures import ensure_cylindrical_uvs, ensure_metric_uvs
from .composer import AssemblyPlan, origin_component_id
from .material_packs import TextureAssignment, verify_texture_targets
from .render_profiles import select_render_profile
from .shots import QualityShot
from .surfaces import get_surface

if TYPE_CHECKING:
    from .asset_spec import AssetSpec


def _inspection_shots(plan: AssemblyPlan, spec: AssetSpec) -> tuple[QualityShot, ...]:
    lo, hi = plan.bounds_min, plan.bounds_max
    width, depth, height = tuple(hi[i] - lo[i] for i in range(3))
    span = max(width, depth, height, 0.1)
    cx, cy, cz = tuple((lo[i]+hi[i])/2 for i in range(3))
    full_scale = max(0.4, width*1.7, height*2.05, depth*1.8)
    primary_id = next((item.id for item in reversed(spec.components)
                       if not item.type.startswith("detail.")), None)
    last = (next(part for part in reversed(plan.instances) if part.name == primary_id)
            if primary_id is not None else plan.instances[-1])
    # The detail view targets the last authored primary piece, not a random
    # camera direction or a radial stud at the edge of the asset.
    coords = tuple(tuple(last.origin[a] + vertex[0]*last.axis_x[a] +
                         vertex[1]*last.axis_y[a] + vertex[2]*last.axis_z[a]
                         for a in range(3)) for vertex in last.mesh.vertices)
    detail_center = tuple((min(p[a] for p in coords)+max(p[a] for p in coords))/2 for a in range(3))
    detail_span = max(max(p[a] for p in coords)-min(p[a] for p in coords) for a in range(3))
    available = {
        "hero": QualityShot("hero", (cx+span*1.2, cy-span*1.6, cz+span*0.8),
                            (cx, cy, cz), full_scale*1.3),
        "front": QualityShot("front", (cx, cy-span*2.0, cz+span*0.12),
                             (cx, cy, cz), full_scale),
        "side": QualityShot("side", (cx+span*2.0, cy, cz+span*0.12),
                            (cx, cy, cz), full_scale),
        "detail": QualityShot("detail", (detail_center[0]+span*0.4,
                                       detail_center[1]-span*0.9,
                                       detail_center[2]+span*0.3),
                              detail_center, max(0.15, detail_span*2.3)),
    }
    return tuple(available[name] for name in spec.quality.views)


def build_assembly_scene(spec: AssetSpec, plan: AssemblyPlan, destination: Path, *,
                         texture_packs: tuple[TextureAssignment, ...] = (),
                         bpy_module: Any = None) -> dict[str, Any]:
    """Render and export an immutable revision through Kumiki's shared contract."""
    if bpy_module is None:
        raise RuntimeError("Blender bpy is required to build an AssetSpec")
    bpy = bpy_module
    if tuple(bpy.app.version) < (4, 2, 0):
        raise RuntimeError("Blender 4.2 or newer is required")
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Output revision already exists: {destination}")

    cinematic = spec.quality.preset == "cinematic-hero"
    width, height, samples = ((1920, 1080, 96) if cinematic else (1440, 900, 48))
    profile = select_render_profile(width, height, samples,
                                    mode="smoke" if spec.quality.preset == "smoke" else None)
    scene = create_model_scene(bpy, spec.asset_id)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = profile.samples
    scene.render.resolution_x = profile.width
    scene.render.resolution_y = profile.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"

    model = bpy.data.collections.new("Composer / portable GLB")
    scene.collection.children.link(model)
    stage = bpy.data.collections.new("Studio / excluded from GLB")
    scene.collection.children.link(stage)
    verify_texture_targets(list(spec.materials), texture_packs)
    assigned = {item.target: item.pack for item in texture_packs}
    if any(spec.materials[role].preset == "optical-glass" for role in assigned):
        raise RuntimeError("Texture packs cannot replace optical-glass transmission materials")
    palette = {role: (create_texture_pack_material(bpy, assigned[role], role)
                      if role in assigned else
                      _material(bpy, get_surface(item.preset, color=item.color,
                                                 roughness=item.roughness)))
               for role, item in spec.materials.items()}
    from mathutils import Matrix  # Blender-only dependency.

    component_type = {part.id: part.type for part in spec.components}
    cache: dict[tuple[int, str], Any] = {}
    objects: list[Any] = []
    for instance in plan.instances:
        role = instance.material
        key = (id(instance.mesh), role)
        component_id = origin_component_id(instance.name, spec.components)
        if component_id is None:
            raise RuntimeError(f"Unregistered generated component: {instance.name}")
        kind = component_type[component_id]
        if key not in cache:
            mesh = bpy.data.meshes.new(f"Geometry / {instance.name}")
            mesh.from_pydata(instance.mesh.vertices, [], instance.mesh.faces)
            mesh.update()
            tile_size = assigned[role].tile_size_m if role in assigned else 1.5
            if kind in ("solid.lathe", "solid.hollow-lathe", "mechanical.rod",
                        "detail.radial-studs", "detail.linear-fasteners"):
                ensure_cylindrical_uvs(mesh, tile_size_m=tile_size)
                # The worker intentionally has no original Pydantic inputs:
                # distinguish six-sided hex heads from curved buttons by the
                # compiled cap topology, preserving crisp hex facets.
                smooth_sides = (kind != "detail.linear-fasteners" or
                                any(len(face.vertices) >= 16 for face in mesh.polygons))
                for polygon in mesh.polygons:
                    if len(polygon.vertices) == 4 and smooth_sides:
                        polygon.use_smooth = True
            else:
                ensure_metric_uvs(mesh, tile_size_m=tile_size)
                if kind == "solid.rounded-box":
                    for polygon in mesh.polygons:
                        polygon.use_smooth = True
            mesh.materials.append(palette[role])
            cache[key] = mesh
        obj = bpy.data.objects.new(instance.name, cache[key])
        model.objects.link(obj)
        obj.matrix_world = Matrix(((instance.axis_x[0], instance.axis_y[0], instance.axis_z[0], instance.origin[0]),
                                   (instance.axis_x[1], instance.axis_y[1], instance.axis_z[1], instance.origin[1]),
                                   (instance.axis_x[2], instance.axis_y[2], instance.axis_z[2], instance.origin[2]),
                                   (0, 0, 0, 1)))
        obj["kumiki_part"] = component_id
        obj["kumiki_component_type"] = kind
        obj["kumiki_material_role"] = role
        if kind not in ("detail.radial-studs", "detail.linear-fasteners", "solid.rounded-box"):
            # Small real bevels give machined edges highlight response, without
            # smoothing the fasteners away or inflating a distant LOD.
            dimensions = [max(v[i] for v in instance.mesh.vertices)-
                          min(v[i] for v in instance.mesh.vertices) for i in range(3)]
            bevel = obj.modifiers.new("Light-catching manufactured edges", "BEVEL")
            bevel.width = min(0.004, max(0.0002, min(dimensions)*0.12))
            bevel.segments = 3 if cinematic else 2
            obj.modifiers.new("Weighted craft normals", "WEIGHTED_NORMAL")
        elif kind == "solid.rounded-box":
            # Preserve broad planar highlights while the real fillet meshes
            # interpolate smoothly around their narrow edge/corner bands.
            obj.modifiers.new("Weighted fillet normals", "WEIGHTED_NORMAL")
        objects.append(obj)

    lo, hi = plan.bounds_min, plan.bounds_max
    span = max((hi[i]-lo[i] for i in range(3)), default=1.0)
    cx, cy = ((lo[i]+hi[i])/2 for i in (0, 1))
    floor = _material(bpy, get_surface("studio-matte", color="#32363A", roughness=0.93))
    _create_box(bpy, stage, "Studio / neutral ground", (span*4, span*4, max(0.025, span*0.03)),
                (cx, cy, lo[2]-max(0.018, span*0.017)), floor)
    world = bpy.data.worlds.new("Kumiki / product studio")
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.12, 0.14, 0.18, 1)
    background.inputs["Strength"].default_value = 0.24
    scene.world = world
    light_scale = min(10.0, max(0.01, span*span*0.30))
    for label, position, energy, size in (
        ("Key / broad softbox", (cx-span, cy-span*1.2, hi[2]+span), 550, span*1.25),
        ("Rim / contour", (cx+span, cy+span*0.85, hi[2]+span*0.7), 700, span*0.9),
        ("Fill / soft", (cx+span*0.5, cy-span*1.2, hi[2]+span*0.4), 240, span*1.2),
    ):
        data = bpy.data.lights.new(label, type="AREA")
        data.energy = energy*light_scale
        data.shape = "DISK"
        data.size = max(0.2, size)
        obj = bpy.data.objects.new(label, data)
        stage.objects.link(obj)
        obj.location = position
        aim_at(obj, ((lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2))

    camera_data = bpy.data.cameras.new("Camera / component inspection")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    stage.objects.link(camera)
    metadata = {
        "asset_id": spec.asset_id,
        "generator": "kumiki-blender/component-composer-v2",
        "status": "procedural-assembly / visual-review-pending",
        "provenance": "Original declarative component geometry; not a real-world reference scan",
        "recipe_sha256": plan.recipe_sha256,
        "render_profile": {"name": profile.name, "width": profile.width,
                           "height": profile.height, "samples": profile.samples},
        "quality_preset": spec.quality.preset,
        "component_types": dict(plan.named_parts),
        "named_parts": dict(plan.named_parts),
        "mesh_data_blocks": len(cache),
        "instances": len(plan.instances),
        "units": "meters",
        "native_axes": "Blender Z-up; GLB exported with glTF-compatible axes",
        "bounds_m": [plan.bounds_min, plan.bounds_max],
    }
    if texture_packs:
        metadata["texture_packs"] = [item.to_manifest() for item in texture_packs]
    return export_revision(bpy, scene, camera, objects, _inspection_shots(plan, spec),
                           destination, samples=profile.samples, metadata=metadata)
