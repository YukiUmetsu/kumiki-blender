"""Synthetic Blender 4.5 smoke: source-verified geodesic landmark fit on a cage.

Runs a real native Blender worker on a separately saved .blend; uses system
Python only for source-image provenance validation (Pillow is not bundled
with Blender Python on all machines).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]


def _write_png(path):
    image=bpy.data.images.new("registered-source",width=1600,height=1200,alpha=False)
    try:
        image.file_format="PNG"
        image.filepath_raw=str(path)
        image.save()
    finally:
        bpy.data.images.remove(image)


def _python_cli(args):
    command=shutil.which("python3") or shutil.which("python")
    if command is None:
        raise RuntimeError("System Python is required for source-image provenance validation")
    env=os.environ.copy()
    env["PYTHONPATH"]=str(ROOT/"src")+os.pathsep+env.get("PYTHONPATH","")
    script=(
        "from kumiki_blender.character_build.landmark_triangulation import triangulate_landmarks;"
        "from kumiki_blender.character_build.landmark_fit_plan import save_landmark_fit_plan;"
        "from pathlib import Path;"
        "triangulate_landmarks(Path(__import__('sys').argv[1]),Path(__import__('sys').argv[2]));"
        "save_landmark_fit_plan(Path(__import__('sys').argv[3]),Path(__import__('sys').argv[4]),"
        "Path(__import__('sys').argv[5]),Path(__import__('sys').argv[6]))"
    )
    run=subprocess.run([command,"-c",script]+[str(p) for p in args],env=env,
                        text=True,capture_output=True,timeout=90)
    if run.returncode:
        raise RuntimeError(f"System Python landmark setup failed: {run.stderr[-3500:]}")


def main():
    with tempfile.TemporaryDirectory(prefix="kumiki-geodesic-landmark-") as tmp:
        folder=Path(tmp)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=13,y_subdivisions=13,
                                        size=.12,location=(0,0,.14))
        patch=bpy.context.object
        patch.name="FacePatch"
        index=min((v.index for v in patch.data.vertices),
                   key=lambda i: patch.data.vertices[i].co.length_squared)
        anchor=patch.matrix_world @ patch.data.vertices[index].co
        target=anchor+Vector((0,0,.0024))
        source_count=len(patch.data.vertices)
        source_uv=[uv.name for uv in patch.data.uv_layers]
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=10,
                                             radius=.02,location=(.28,0,.15))
        other=bpy.context.object
        other.name="UnrelatedNose"
        nose_vertices=[tuple(v.co) for v in other.data.vertices]
        source=folder/"original.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        source_sha=hashlib.sha256(source.read_bytes()).hexdigest()

        references=folder/"refs"
        references.mkdir()
        for view in ("front","right"):
            _write_png(references/f"{view}.png")
        height=.35
        # Exact canonical camera projections for horizontal front and right:
        # front: x and z; right: y and z, both at 1000 pixels/character height.
        col_front=800+target.x/(height/1000)
        col_right=800-target.y/(height/1000)
        row=1100-target.z/(height/1000)
        landmark_source={
            "schema_version":1,"character_id":"face-demo","height_m":height,
            "registered_views":{
                view:{"file":f"refs/{view}.png","registration":"original-aligned"}
                for view in ("front","right")
            },
            "landmarks":[{
                "id":"lid.center","max_reprojection_error_px":.2,
                "observations":[
                    {"view":"front","xy_px":[col_front,row],"confidence":1},
                    {"view":"right","xy_px":[col_right,row],"confidence":1},
                ]
            }]
        }
        source_ref=folder/"source-landmarks.json"
        source_ref.write_text(json.dumps(landmark_source))
        constraints=folder/"fit.json"
        constraints.write_text(json.dumps({
            "schema_version":1,"mesh":"FacePatch",
            "expected_vertex_count":source_count,
            "max_displacement_m":.005,"max_relative_edge_change":.25,
            "protected_vertices":[0,12,156,168],
            "pins":[{"id":"lid.center","source_vertex":index,"radius_m":.044}],
        }))
        landmark_dir=folder/"landmarks"
        plan_dir=folder/"fit-plan"
        _python_cli([source_ref,landmark_dir,source,
                     landmark_dir/"landmarks.json",constraints,plan_dir])
        plan=json.loads((plan_dir/"landmark-fit-plan.json").read_text())
        output=folder/"fitted"
        args=folder/"worker-args.json"
        args.write_text(json.dumps({"plan":plan,"output":str(output)}))
        script=ROOT/"src/kumiki_blender/character_build/blender_landmark_fit.py"
        proc=subprocess.run([
            bpy.app.binary_path,"--background","--python-exit-code","1",
            str(source),"--python",str(script),"--",str(args)
        ],capture_output=True,text=True,timeout=100)
        if proc.returncode or not (output/"report.json").is_file():
            print(proc.stdout[-12000:],file=sys.stderr)
            print(proc.stderr[-12000:],file=sys.stderr)
            raise RuntimeError(f"Native geodesic fit worker failed: {proc.returncode}")
        report=json.loads((output/"report.json").read_text())
        assert report["status"]=="geodesic-landmark-candidate-visual-qa-required"
        assert report["changed_vertices"]>1
        assert report["changed_vertices"]<source_count
        assert report["max_displacement_m"]<=.005
        assert report["flipped_or_degenerate_faces"]==0
        assert report["unrelated_meshes_unchanged"] is True
        assert report["original_materials_and_uvs_unchanged"] is True
        assert (output/"landmark-fitted.obj").is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
        bpy.ops.wm.open_mainfile(filepath=str(output/"landmark-fitted.blend"))
        obj=bpy.data.objects["FacePatch"]
        placed=obj.matrix_world @ obj.data.vertices[index].co
        assert (placed-target).length<1e-6,((placed-target).length,tuple(placed),tuple(target))
        assert [uv.name for uv in obj.data.uv_layers]==source_uv
        assert [tuple(v.co) for v in bpy.data.objects["UnrelatedNose"].data.vertices]==nose_vertices
        print(json.dumps({
            "status":"native-source-geodesic-fit-verified",
            "blender":bpy.app.version_string,
            "landmark_error_m":round((placed-target).length,10),
            "changed_vertices":report["changed_vertices"],
            "original_source_sha256_unchanged":True,
            "unrelated_meshes_unchanged":True,
        },indent=2))


main()
