import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.blender_scene import _beam_geometry, _box_geometry, _gusset_geometry, build_scene
from kumiki_blender.blender_portico import build_portico_scene
from kumiki_blender.lattice import Beam, make_lattice_tower
from kumiki_blender.portico import make_portico_study
from kumiki_blender.recipe import load_recipe


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/lattice-observation-tower.json"


class BlenderSceneContractTests(unittest.TestCase):
    def test_gusset_mesh_is_oriented_toward_the_outside_of_the_tower(self):
        recipe = load_recipe(FIXTURE)
        gusset = make_lattice_tower(recipe).gussets[0]
        vertices, faces = _gusset_geometry(gusset)
        self.assertEqual(len(vertices), 8)
        self.assertEqual(len(faces), 6)
        a, b, c = [vertices[index] for index in faces[1][:3]]
        ab, ac = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        normal = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
        self.assertGreater(sum(normal[i] * gusset.normal[i] for i in range(3)), 0)

    def test_beam_is_closed_with_outward_side_normals(self):
        beam = Beam("sample", (0, 0, 0), (0, 0, 10), 1, "leg")
        vertices, faces = _beam_geometry(beam)
        self.assertEqual(len(faces), 18)
        for face in faces[:16]:
            a, b, c = [vertices[index] for index in face[:3]]
            ab, ac = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
            normal = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
            self.assertGreater(normal[0] * a[0] + normal[1] * a[1], 0)

    def test_box_faces_have_outward_normals(self):
        center = (1.0, -2.0, 3.0)
        vertices, faces = _box_geometry((2.0, 4.0, 6.0), center)
        for face in faces:
            a, b, c = [vertices[index] for index in face[:3]]
            ab, ac = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
            normal = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
            face_center = tuple(sum(vertices[index][axis] for index in face) / len(face) for axis in range(3))
            outward = sum(normal[i] * (face_center[i] - center[i]) for i in range(3))
            self.assertGreater(outward, 0, f"Inward normal for face {face}")

    def test_missing_blender_context_does_not_create_files(self):
        recipe = load_recipe(FIXTURE)
        plan = make_lattice_tower(recipe)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "uncreated"
            with self.assertRaisesRegex(RuntimeError, "Blender"):
                build_scene(recipe, plan, output, bpy_module=None)
            self.assertFalse(output.exists())

    def test_portico_requires_blender_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "uncreated"
            with self.assertRaisesRegex(RuntimeError, "Blender"):
                build_portico_scene(make_portico_study(), output, bpy_module=None)
            self.assertFalse(output.exists())

    @unittest.skipUnless(shutil.which("blender"), "Blender executable unavailable in this environment")
    def test_real_blender_writes_editable_source_glb_preview_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "r01"
            result = subprocess.run([
                shutil.which("blender"), "--background", "--factory-startup", "--python-exit-code", "1",
                "--python", str(ROOT / "scripts/blender_entry.py"), "--",
                "build", "--recipe", str(FIXTURE), "--output", str(output),
            ], capture_output=True, text=True, timeout=300, check=False)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            for expected in ("source.blend", "model.glb", "preview.png", "manifest.json",
                             "previews/front.png", "previews/side.png", "previews/detail.png"):
                self.assertGreater((output / expected).stat().st_size, 0)

    @unittest.skipUnless(shutil.which("blender"), "Blender executable unavailable in this environment")
    def test_real_blender_portico_exports_meshes_with_instanced_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "portico-r01"
            result = subprocess.run([
                shutil.which("blender"), "--background", "--factory-startup", "--python-exit-code", "1",
                "--python", str(ROOT / "scripts/blender_entry.py"), "--",
                "build-portico", "--bays", "2", "--output", str(output),
            ], capture_output=True, text=True, timeout=360, check=False)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            for expected in ("source.blend", "model.glb", "preview.png", "manifest.json",
                             "previews/front.png", "previews/detail.png"):
                self.assertGreater((output / expected).stat().st_size, 0)
            import json

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["named_parts"]["arch_stones"], 22)
            self.assertEqual(manifest["named_parts"]["columns"], 3)


if __name__ == "__main__":
    unittest.main()
