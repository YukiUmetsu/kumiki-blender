# Red vintage biplane — original procedural model and animation rig

The `red-vintage-biplane` generator interprets the user-provided **top, bottom,
front, rear and side** illustration into an original, editable, metrically
consistent Blender aircraft. It does **not** claim to identify or reproduce a
measured historic airframe. No externally downloaded models or reference
textures are included; all geometry and material maps are generated locally.

## Reference breakdown

| Reference feature | 3D interpretation | Why it matters |
|---|---|---|
| Long, rectangular fabric wings with rounded wingtips | Cambered closed upper/lower skins, elliptical wing tips and 74 raised rib tapes | Captures fabric tension instead of flat slabs |
| One upper and one lower wing, upper slightly ahead | 9.66 m upper and 9.32 m lower span, 1.10 m vertical separation | Preserves the characteristic biplane stagger |
| Wooden vertical struts and X-shaped black tension rigging | 12 interplane/cabane struts plus crossing flying wires in each bay | Wires are actual slender 3D solids, not screen-space strokes |
| Rounded engine cowling, dark grille, two-blade wood propeller | Tapered cowl, individual grille bars, rivets, exhausts, and varnished sculpted propeller blades | Distinct metals, timber and red painted surfaces |
| Open single cockpit immediately behind upper wing | Recessed well, shaped red side fairings, leather coaming/seat, stick and sloped glass windscreen | True open-cockpit silhouette from side/top |
| Narrowing red fuselage and large swept fin | 3D lofted front/tail sections, separately hinged rudder, stabilizers and elevators | Convincing side profile and articulating tail surfaces |
| Fixed two-wheel undercarriage | Real toroidal rubber tires, lacquered red hubs with brass trim/rivets, angled wooden struts and axle wire bracing | Visible axle-to-fuselage structural load path; asymmetric hub balance weights expose wheel spin |

Coordinates: **Z up, nose along −Y, wings along ±X**. The tail ends near Y=+3.69 m and the propeller extends to −Y≈3.59 m. The model sits on its two wheels at roughly Z=0; dimensions are artistic estimates based on the illustration's relative proportions.

### Materials and finish

- Fabric wing red: `#AE211C`, satin with subtly raised, separately named rib tape.
- Fuselage enamel: `#AC251D`, smoother and glossier than fabric.
- Deep oxblood hardware: `#78211D`; varnished hardwood struts: `#A56A3D`.
- Two-blade propeller: `#C58F55` with a reusable packed timber PBR map.
- Metallic grille, brass rivets and hubs, black rubber, dark leather and tinted glass.

The reusable Kumiki material generator embeds deterministic albedo, roughness and normal textures where the material supports them. The studio floor/lights are excluded from the GLB.

## Rig

The native `.blend` contains `RIG / aircraft-root` and nine independently named transform controls. Hinges are placed at each moving surface's **actual geometry**, rather than sharing one incorrect pivot for different-height wings.

| Control suffix | Rotation axis | Intended use |
|---|---|---|
| `propeller` | Local Y | Eight propeller rotations in a 96-frame loop |
| `upper-left-aileron`, `upper-right-aileron` | Local X | Opposed wing roll deflection |
| `lower-left-aileron`, `lower-right-aileron` | Local X | Independent lower wing hinges, matching roll direction |
| `elevator` | Local X | Paired tail surfaces pitch together |
| `rudder` | Local Z | Tail yaw |
| `left-wheel`, `right-wheel` | Local X | Wheel rolling motion around the axle |

All geometry is a child of the root or its named control. The default `flight-loop` is 97 keyed endpoints (96-frame interval), 24 fps. It is only a **mechanical demonstration**; it does not purport to simulate flight dynamics. Nine control animations are exported to GLB, while the editable `.blend` retains the object/action hierarchy.

To inspect the spinning components in Blender, select `RIG / propeller` or
`RIG / left-wheel` / `RIG / right-wheel`, then scrub the timeline. The engine
propeller completes eight revolutions and each wheel three revolutions over
frames 1–97. Their axes are distinct: the propeller rotates about **Y**;
both independent wheel assemblies rotate about their **X** axles. The tires,
red covers, trim rings, brass rivets, and asymmetric balance clips are actual
children of the respective wheel controls; the struts and fixed axle stay
still. The rig checker samples intermediate frames and compares actual mesh
world positions, not merely animation metadata.

## Run

```bash
PYTHONPATH=src python3 -m kumiki_blender plan-biplane

KUMIKI_RENDER_PROFILE=smoke blender --background --factory-startup \
  --python-exit-code 1 --python scripts/blender_entry.py -- \
  build-biplane --output .review/biplane/r01

PYTHONPATH=src python3 -m kumiki_blender audit-revision --root .review/biplane/r01
PYTHONPATH=src python3 -m kumiki_blender audit --model .review/biplane/r01/model.glb --require-pbr-textures
```

Choose a new revision path (`r02`, `r03`, …) for every build; the exporter deliberately refuses overwriting existing revisions.

Output: `source.blend`, `model.glb`, `preview.png`, six inspection views (`front`, `side`, `top`, `rear`, `cockpit`, plus the hero), and `manifest.json`. Remove the `KUMIKI_RENDER_PROFILE=smoke` variable for full-resolution 1440×810, 72-sample Cycles images. The tests validate manifold meshes and control pivots, but **human review of the actual rendered previews remains necessary**.

### Limitations

- The illustration is not a dimensioned aircraft blueprint; span, airfoil, engine and gear are reconstructed from visual proportions.
- Textured weathering is generic first-party procedural PBR, not individually painted historic fabric or photogrammetry.
- The current rig includes mechanical articulation, not inverse kinematics for landing-gear compression or cloth deformation.
- This object can be animated in Blender/Kumiki-UI and GLB consumers, but an external renderer must load and play the relevant glTF animation clips explicitly.
