"""Real camera reprojection and open-boundary QA on small synthetic meshes."""

import contextlib
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.surface_coverage import (
    _boundary_edges, _load_triangular_ply, audit_surface_coverage, plan_surface_coverage,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SurfaceCoverageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        sparse = self.root / "sparse"
        self.dense = self.root / "dense"
        self.workspace = self.dense / "workspace"
        self.workspace.mkdir(parents=True)
        (self.workspace / "images").mkdir()
        model = sparse / "undistorted/text-model"
        photos = sparse / "undistorted/images"
        model.mkdir(parents=True)
        photos.mkdir()
        (model / "cameras.txt").write_text("1 PINHOLE 64 64 35 35 32 32\n")
        (model / "images.txt").write_text("".join(
            f"{i+1} 1 0 0 0 {i * .2} 0 4 1 view-{i:02d}.png\n\n"
            for i in range(3)))
        entries = []
        mask_records = []
        for i in range(3):
            name = f"view-{i:02d}.png"
            photo = photos / name
            Image.new("RGB", (64, 64), (180, 160, 130)).save(photo)
            entries.append({"filename": name, "sha256": sha(photo), "dimensions": [64, 64]})
            mask = self.workspace / "images" / (name + ".mask.png")
            image = Image.new("L", (64, 64), 255)
            # The third view excludes the right half of the object, while
            # the first two images still observe it.
            if i == 2:
                image.paste(0, (33, 0, 64, 64))
            else:
                image.paste(0, (0, 0, 3, 64))
            image.save(mask)
            mask_records.append({"filename": name, "sha256": sha(mask)})
        (sparse / "undistorted-reference-index.json").write_text(json.dumps({"images": entries}))
        (sparse / "reconstruction-report.json").write_text(json.dumps({
            "status": "candidate / review-required", "registered_images": 3}))
        self.mesh = self.workspace / "scene_dense_mesh.ply"
        self.vertices = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        self.faces = [(0, 1, 2), (0, 2, 3)]
        self.write_mesh("ascii")
        self.report = self.dense / "dense-report.json"
        self.mask_records = mask_records
        self.write_report(sparse)

    def write_mesh(self, fmt):
        extra_vertex_fields = ("property uchar red\nproperty uchar green\nproperty uchar blue\n"
                               if fmt == "binary" else "")
        head = ("ply\nformat " + ("ascii 1.0" if fmt == "ascii" else "binary_little_endian 1.0")
                + "\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
                  + extra_vertex_fields +
                  "element face 2\nproperty list uchar uint vertex_indices\nend_header\n")
        if fmt == "ascii":
            body = "".join(" ".join(str(v) for v in xyz) + "\n" for xyz in self.vertices)
            body += "".join("3 " + " ".join(str(v) for v in face) + "\n" for face in self.faces)
            self.mesh.write_text(head + body)
        else:
            vertices = b"".join(struct.pack("<fffBBB", *point, 180, 130, 80)
                                for point in self.vertices)
            faces = b"".join(struct.pack("<BIII", 3, *face) for face in self.faces)
            self.mesh.write_bytes(head.encode() + vertices + faces)

    def write_report(self, sparse):
        self.report.write_text(json.dumps({
            "status": "dense-mesh-candidate / artistic-and-rights-review-required",
            "source_revision": str(sparse), "registered_images": 3,
            "mesh": {"ply": "workspace/scene_dense_mesh.ply", "sha256": sha(self.mesh),
                     "vertices": 4, "faces": 2}, "rectified_masks": self.mask_records,
        }))

    def test_ascii_and_binary_meshes_report_open_edges_and_camera_support(self):
        for fmt in ("ascii", "binary"):
            with self.subTest(fmt=fmt):
                self.write_mesh(fmt)
                self.write_report(self.root / "sparse")
                output = self.root / f"qa-{fmt}"
                report = audit_surface_coverage(self.dense, output, min_views=3)
                self.assertIn("coverage-gap-detected", report["status"])
                self.assertGreater(report["possible_view_support"]["weak_fraction"], 0)
                self.assertEqual(report["sampled_faces"], 2)
                self.assertEqual(report["topology"]["boundary_edges"], 4)
                self.assertEqual(report["topology"]["nonmanifold_edges"], 0)
                self.assertTrue((output / "view-support-points.ply").is_file())
                self.assertTrue((output / "open-boundary-points.ply").is_file())
                self.assertFalse(report["photographic_fidelity_verified"])
                self.assertFalse(report["distribution_allowed"])
                self.assertEqual(len(report["per_camera"]), 3)

    def test_mask_aware_upper_bound_and_no_mask_override(self):
        with_masks = audit_surface_coverage(
            self.dense, self.root / "masked", min_views=3)
        without_masks = audit_surface_coverage(
            self.dense, self.root / "unmasked", min_views=3, use_masks=False)
        self.assertGreater(with_masks["possible_view_support"]["weak_fraction"],
                           without_masks["possible_view_support"]["weak_fraction"])
        self.assertEqual(without_masks["possible_view_support"]["three_or_more"], 2)
        self.assertFalse(without_masks["foreground_masks_applied"])

    def test_closed_mesh_has_no_open_boundaries(self):
        positions = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        _points, topology = _boundary_edges(positions, faces, max_points=100, np=np)
        self.assertEqual(topology["boundary_edges"], 0)

    def test_preflight_is_nonmutating_and_rejects_tampering(self):
        out = self.root / "planned"
        record = plan_surface_coverage(self.dense, out)
        self.assertEqual(record["registered_cameras"], 3)
        self.assertFalse(out.exists())
        with self.assertRaisesRegex(ReferenceError, "max_samples"):
            plan_surface_coverage(self.dense, out, max_samples=True)
        with self.assertRaisesRegex(ReferenceError, "max_mesh_faces"):
            plan_surface_coverage(self.dense, out, max_mesh_faces=999)
        original = self.mesh.read_bytes()
        self.mesh.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ReferenceError, "changed after"):
            plan_surface_coverage(self.dense, out)
        self.mesh.write_bytes(original)
        mask = self.workspace / "images/view-01.png.mask.png"
        original_mask = mask.read_bytes()
        mask.write_bytes(original_mask + b"tampered")
        with self.assertRaisesRegex(ReferenceError, "mask changed"):
            plan_surface_coverage(self.dense, out)
        mask.write_bytes(original_mask)

    def test_cli_plan_and_audit_with_private_immutable_output(self):
        out = self.root / "cli-audit"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["plan-dense-surface-coverage", "--dense-revision",
                                   str(self.dense), "--output", str(out)]), 0)
            self.assertFalse(out.exists())
            self.assertEqual(main(["audit-dense-surface-coverage", "--dense-revision",
                                   str(self.dense), "--output", str(out), "--min-views", "3"]), 0)
        self.assertTrue((out / "surface-coverage-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
