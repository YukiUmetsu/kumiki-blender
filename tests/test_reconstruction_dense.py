"""Dense MVS command boundary, immutable inputs, and geometry gates.

These fake executables validate orchestration; actual reconstruction quality
must still be reviewed on a rights-cleared overlapping-view dataset.
"""

import hashlib
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.dense import (
    DenseSettings, _ply_counts, build_dense_reconstruction, plan_dense_reconstruction,
)
from kumiki_blender.reconstruction.references import ReferenceError

FAKE_MVS = '''#!/usr/bin/env python3
import os
import sys
from pathlib import Path

app = Path(sys.argv[0]).name
args = sys.argv[1:]
def flag(name):
    return Path(args[args.index(name) + 1])
if os.environ.get("FAIL_MVS_STAGE") == app:
    print("intentionally failing " + app)
    sys.exit(23)
if app == "InterfaceCOLMAP":
    assert (flag("--input-file") / "images").is_dir()
elif app == "DensifyPointCloud":
    assert flag("--input-file").name == "scene.mvs"
elif app == "ReconstructMesh":
    assert flag("--input-file").name == "scene_dense.mvs"
    out = flag("--output-file")
    (out.parent / "scene_dense_mesh.ply").write_bytes(
        b"ply\\nformat ascii 1.0\\nelement vertex 2000\\n"
        b"property float x\\nelement face 1600\\nend_header\\n" + b"body"
    )
elif app == "RefineMesh":
    assert flag("--input-file").name == "scene_dense.mvs"
    assert flag("--mesh-file").name == "scene_dense_mesh.ply"
    (flag("--output-file").parent / "scene_dense_mesh_refined.ply").write_bytes(
        b"ply\\nformat ascii 1.0\\nelement vertex 2400\\n"
        b"element face 1900\\nend_header\\n" + b"body"
    )
elif app == "TextureMesh":
    assert flag("--input-file").name == "scene_dense.mvs"
    assert flag("--mesh-file").name in ("scene_dense_mesh_refined.ply", "scene_dense_mesh.ply")
    assert flag("--export-type") == Path("obj")
    if "SKIP_MVS_ATLAS" not in os.environ:
        (flag("--output-file").parent / "scene_dense_mesh_refined_texture_0.png").write_bytes(b"fake atlas data")
else:
    sys.exit(99)
if os.environ.get("SKIP_MVS_OUTPUT") != app:
    flag("--output-file").write_bytes(b"synthetic MVS output")
'''

FAKE_COLMAP = '''#!/usr/bin/env python3
import os
import sys
from pathlib import Path

stage = sys.argv[1]
args = sys.argv[2:]
def flag(name):
    return Path(args[args.index(name)+1])
if os.environ.get("FAIL_MVS_STAGE") == stage:
    sys.exit(18)
if stage == "patch_match_stereo":
    assert flag("--workspace_path").is_dir()
    assert flag("--workspace_format") == Path("COLMAP")
elif stage == "stereo_fusion":
    flag("--output_path").write_bytes(b"fake fused point cloud")
elif stage == "poisson_mesher":
    flag("--output_path").write_bytes(
        b"ply\\nformat binary_little_endian 1.0\\nelement vertex 3000\\n"
        b"element face 2000\\nend_header\\n" + b"binary"
    )
elif stage == "mesh_texturer":
    folder = flag("--output_path")
    folder.mkdir()
    (folder / "mesh.obj").write_bytes(b"fake textured mesh")
else:
    sys.exit(99)
'''


class DenseMvsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sparse = self.root / "sparse-r01"
        rectified = self.sparse / "undistorted"
        images = rectified / "images"
        images.mkdir(parents=True)
        (rectified / "sparse").mkdir()
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (rectified / "sparse" / name).write_bytes(b"fake COLMAP test data")
        (rectified / "stereo").mkdir()
        entries = []
        for n in range(10):
            filename = f"view-{n:02d}.jpg"
            raw = f"first-party-test-frame-{n}".encode()
            (images / filename).write_bytes(raw)
            entries.append({"filename": filename,
                            "sha256": hashlib.sha256(raw).hexdigest()})
        (self.sparse / "undistorted-reference-index.json").write_text(json.dumps({"images": entries}))
        (self.sparse / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required", "asset_id": "test-statue",
            "registered_images": 10, "staged_independent_images": 12,
            "undistorted_images": 10, "manifest_sha256": "a" * 64,
        }))
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh",
                     "RefineMesh", "TextureMesh", "colmap"):
            executable = self.bin / name
            executable.write_text(FAKE_COLMAP if name == "colmap" else FAKE_MVS)
            executable.chmod(0o755)
        self.executables = {name: str(self.bin / name) for name in
                            ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh",
                             "RefineMesh", "TextureMesh", "colmap")}

    def test_cpu_openmvs_dense_mesh_and_texturing_are_checkpointed(self):
        output = self.root / "dense-r01"
        report = build_dense_reconstruction(self.sparse, output, DenseSettings(),
                                            executables=self.executables)
        self.assertEqual(report["status"], "dense-mesh-candidate / artistic-and-rights-review-required")
        self.assertFalse(report["distribution_allowed"])
        self.assertEqual(report["mesh"]["ply"], "workspace/scene_dense_mesh_refined.ply")
        self.assertEqual(report["mesh"]["vertices"], 2400)
        self.assertEqual(report["mesh"]["faces"], 1900)
        self.assertEqual(report["raw_mesh"]["faces"], 1600)
        self.assertEqual(report["refined_face_retention"], round(1900 / 1600, 5))
        self.assertEqual(len(report["mesh"]["sha256"]), 64)
        self.assertEqual([step["stage"] for step in report["commands"]],
                         ["interface", "densify", "mesh", "refine", "texture"])
        densify = report["commands"][1]["argv"]
        self.assertEqual(densify[densify.index("--resolution-level") + 1], "0")
        self.assertEqual(densify[densify.index("--max-resolution") + 1], "3200")
        self.assertEqual(densify[densify.index("--tower-mode") + 1], "0")
        refinement = report["commands"][3]["argv"]
        self.assertEqual(refinement[refinement.index("--decimate") + 1], "1.0")
        self.assertEqual(len(list((output / "logs").glob("*.log"))), 5)
        self.assertEqual(json.loads((output / "dense-report.json").read_text()), report)
        self.assertTrue((output / "workspace/scene_dense_mesh_refined_texture.obj").is_file())
        self.assertEqual(report["texture_images"],
                         ["workspace/scene_dense_mesh_refined_texture_0.png"])
        self.assertEqual(len(report["textured_obj"]["sha256"]), 64)
        self.assertEqual(report["texture_atlases"][0]["sha256"],
                         hashlib.sha256(b"fake atlas data").hexdigest())
        self.assertTrue((self.sparse / "undistorted/images/view-00.jpg").is_file())

    def test_openmvs_mesh_needs_ply_not_optional_mvs_archive(self):
        # OpenMVS 2.4 writes PLY for meshing/refinement, not an .mvs archive.
        fixture = FAKE_MVS.replace(
            'flag("--output-file").write_bytes(b"synthetic MVS output")',
            'flag("--output-file").write_bytes(b"synthetic MVS output") if app in '
            '("InterfaceCOLMAP", "DensifyPointCloud", "TextureMesh") else None')
        for name in ("ReconstructMesh", "RefineMesh", "TextureMesh"):
            (self.bin / name).write_text(fixture)
        result = build_dense_reconstruction(
            self.sparse, self.root / "ply-only", DenseSettings(),
            executables=self.executables)
        self.assertTrue(result["status"].startswith("dense-mesh-candidate"), result)
        self.assertTrue((self.root / "ply-only/workspace/scene_dense_mesh_refined.ply").is_file())

    def test_detail_settings_are_bounded_and_audited(self):
        for changed in ({"resolution_level": -1}, {"resolution_level": True},
                        {"max_resolution": 100}, {"number_views": 1},
                        {"number_views": 100}, {"refine_decimate": 1.2},
                        {"refine_decimate": 0.0}, {"refine_decimate": True}):
            with self.subTest(changed=changed), self.assertRaises(ReferenceError):
                DenseSettings(**changed).validate()
        output = self.root / "dense-quality"
        settings = DenseSettings(resolution_level=1, max_resolution=1600,
                                 number_views=8, texture=False, refine=False)
        report = build_dense_reconstruction(self.sparse, output, settings,
                                            executables=self.executables)
        self.assertEqual(report["resolution_level"], 1)
        self.assertEqual(report["max_resolution"], 1600)
        self.assertEqual(report["number_views"], 8)

    def test_gpu_colmap_always_copies_undistorted_workspace(self):
        output = self.root / "dense-r02"
        report = build_dense_reconstruction(self.sparse, output,
                                            DenseSettings(backend="colmap", refine=False),
                                            executables=self.executables)
        self.assertTrue(report["status"].startswith("dense-mesh-candidate"))
        self.assertEqual(report["mesh"]["faces"], 2000)
        self.assertEqual([step["stage"] for step in report["commands"]],
                         ["patch_match", "fusion", "mesh", "texture"])
        self.assertTrue((output / "workspace/images/view-00.jpg").is_file())
        self.assertFalse((self.sparse / "undistorted/meshed-poisson.ply").exists())

    def test_preflight_refuses_insufficient_coverage_unsafe_revision_and_existing_output(self):
        output = self.root / "output"
        with self.assertRaisesRegex(ReferenceError, r"needs 11\+ views"):
            plan_dense_reconstruction(self.sparse, output,
                                      DenseSettings(min_registered_images=11),
                                      executables=self.executables)
        self.assertFalse(output.exists())
        index_path = self.sparse / "undistorted-reference-index.json"
        original = index_path.read_text()
        index = json.loads(original)
        index["images"][0]["filename"] = "../escape.jpg"
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ReferenceError, "Unsafe or duplicate"):
            plan_dense_reconstruction(self.sparse, output, DenseSettings(),
                                      executables=self.executables)
        self.assertFalse(output.exists())
        index_path.write_text(original)
        output.mkdir()
        (output / "keep").write_text("do not delete")
        with self.assertRaisesRegex(ReferenceError, "already exists"):
            plan_dense_reconstruction(self.sparse, output, DenseSettings(),
                                      executables=self.executables)
        self.assertEqual((output / "keep").read_text(), "do not delete")

    def test_missing_openmvs_binary_fails_without_creating_output(self):
        output = self.root / "no-tools"
        with patch("kumiki_blender.reconstruction.dense.shutil.which", return_value=None):
            with self.assertRaisesRegex(ReferenceError, "Required dense reconstruction executable"):
                plan_dense_reconstruction(self.sparse, output, DenseSettings())
        self.assertFalse(output.exists())

    def test_tampered_rectified_image_and_source_link_are_rejected_before_output(self):
        changed = self.sparse / "undistorted/images/view-00.jpg"
        changed.write_text("tampered")
        with self.assertRaisesRegex(ReferenceError, "changed since reconstruction"):
            plan_dense_reconstruction(self.sparse, self.root / "out",
                                      DenseSettings(), executables=self.executables)
        self.assertFalse((self.root / "out").exists())
        changed.write_bytes(b"first-party-test-frame-0")
        (self.sparse / "undistorted/images/view-01.jpg").unlink()
        (self.sparse / "undistorted/images/view-01.jpg").symlink_to(changed)
        with self.assertRaisesRegex(ReferenceError, "linked outside"):
            plan_dense_reconstruction(self.sparse, self.root / "out",
                                      DenseSettings(), executables=self.executables)

    def test_failed_external_stage_retains_diagnostic_report_and_logs(self):
        with patch.dict(os.environ, {"FAIL_MVS_STAGE": "DensifyPointCloud"}):
            report = build_dense_reconstruction(self.sparse, self.root / "bad",
                                                DenseSettings(), executables=self.executables)
        self.assertEqual(report["status"], "review-required")
        self.assertEqual(report["failed_stage"], "densify")
        self.assertEqual(report["exit_code"], 23)
        self.assertIn("intentionally failing", (self.root / "bad/logs/02-densify.log").read_text())

    def test_success_exit_cannot_hide_missing_mesh_or_texture_output(self):
        with patch.dict(os.environ, {"SKIP_MVS_OUTPUT": "DensifyPointCloud"}):
            report = build_dense_reconstruction(self.sparse, self.root / "missing",
                                                DenseSettings(), executables=self.executables)
        self.assertEqual(report["failed_stage"], "densify_output")
        self.assertFalse(report["status"].startswith("dense-mesh-candidate"))
        with patch.dict(os.environ, {"SKIP_MVS_ATLAS": "1"}):
            textured = build_dense_reconstruction(self.sparse, self.root / "flat",
                                                  DenseSettings(), executables=self.executables)
        self.assertEqual(textured["failed_stage"], "texture_output")
        self.assertEqual(textured["status"], "review-required")

    def test_resume_skips_valid_completed_cpu_stages_and_rechecks_input_hashes(self):
        output = self.root / "resume"
        with patch.dict(os.environ, {"FAIL_MVS_STAGE": "RefineMesh"}):
            failed = build_dense_reconstruction(self.sparse, output, DenseSettings(),
                                                executables=self.executables)
        self.assertEqual(failed["failed_stage"], "refine")
        commands_before = len(failed["commands"])
        repaired = build_dense_reconstruction(
            self.sparse, output, DenseSettings(),
            executables=self.executables, resume=True)
        self.assertTrue(repaired["status"].startswith("dense-mesh-candidate"), repaired)
        self.assertEqual(repaired["resume_skipped_stages"], ["interface", "densify", "mesh"])
        self.assertEqual(len(repaired["commands"]), commands_before + 2)
        with self.assertRaisesRegex(ReferenceError, "incomplete"):
            build_dense_reconstruction(self.sparse, output, DenseSettings(),
                                       executables=self.executables, resume=True)

        broken = self.root / "broken-resume"
        with patch.dict(os.environ, {"FAIL_MVS_STAGE": "DensifyPointCloud"}):
            build_dense_reconstruction(self.sparse, broken, DenseSettings(),
                                       executables=self.executables)
        (broken / "workspace/images/view-00.jpg").write_text("tampered")
        with self.assertRaisesRegex(ReferenceError, "Resume workspace image changed"):
            build_dense_reconstruction(self.sparse, broken, DenseSettings(),
                                       executables=self.executables, resume=True)
        self.assertEqual(json.loads((broken / "dense-report.json").read_text())["failed_stage"],
                         "densify")

    def test_resume_can_skip_unstarted_refinement_for_high_poly_raw_scan(self):
        output = self.root / "raw-resume"
        with patch.dict(os.environ, {"FAIL_MVS_STAGE": "ReconstructMesh"}):
            failed = build_dense_reconstruction(self.sparse, output, DenseSettings(),
                                                executables=self.executables)
        self.assertEqual(failed["failed_stage"], "mesh")
        result = build_dense_reconstruction(self.sparse, output,
                                            DenseSettings(refine=False),
                                            executables=self.executables, resume=True)
        self.assertTrue(result["status"].startswith("dense-mesh-candidate"), result)
        self.assertEqual(result["mesh"]["faces"], 1600)
        self.assertFalse(result["refine"])
        self.assertEqual(result["resume_skipped_stages"], ["interface", "densify"])

    def test_invalid_mesh_fails_even_when_external_commands_return_success(self):
        result = build_dense_reconstruction(self.sparse, self.root / "thin",
                                            DenseSettings(min_mesh_faces=2000, texture=False),
                                            executables=self.executables)
        self.assertEqual(result["failed_stage"], "mesh_validation")
        self.assertEqual(result["mesh"]["faces"], 1900)
        self.assertEqual(_ply_counts(self.root / "thin/workspace/scene_dense_mesh.ply")["vertices"], 2000)

    def test_cli_planning_and_building_use_the_same_contract(self):
        output = self.root / "cli"
        with patch("kumiki_blender.reconstruction.dense.shutil.which",
                   side_effect=lambda name: self.executables.get(name)):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["plan-dense-photos", "--reconstruction", str(self.sparse),
                                       "--output", str(output)]), 0)
                self.assertFalse(output.exists())
                self.assertEqual(main(["build-dense-photos", "--reconstruction", str(self.sparse),
                                       "--output", str(output)]), 0)
        self.assertTrue((output / "workspace/scene_dense_mesh.ply").is_file())


if __name__ == "__main__":
    unittest.main()
