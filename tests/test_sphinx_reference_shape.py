"""Photo-to-original-sculpt warping; no reference images are shipped or read."""

from __future__ import annotations

import contextlib
import io
import json
import math
import unittest
from pathlib import Path

from kumiki_blender.architecture_geometry import audit_mesh
from kumiki_blender.cli import main
from kumiki_blender.egyptian_monuments import _anatomical_sphinx_face, make_monument_study
from kumiki_blender.sphinx_landmarks import LandmarkError, LandmarkView, load_landmark_view
from kumiki_blender.sphinx_reference_shape import (
    deform_sphinx_face, prepare_sphinx_face_fit,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "examples/sphinx-frontal-photo-landmarks.json"
CANDIDATE = ROOT / "examples/sphinx-frontal-procedural-landmarks.json"
Z_SCALE = 3.6/4.6


class SphinxReferenceShapeTests(unittest.TestCase):
    def fit(self):
        return prepare_sphinx_face_fit(load_landmark_view(REFERENCE),
                                       load_landmark_view(CANDIDATE))

    def test_reference_drives_human_feature_translations_not_global_scaling(self):
        fit = self.fit()
        controls = {c.name: c for c in fit.controls}
        self.assertEqual(len(controls), 7)
        self.assertEqual(len(fit.reference_sha256), 64)
        # User's photo places eyes higher, mouth lower and chin higher than
        # the previously rendered procedural head when nemes/crown are aligned.
        self.assertGreater(controls["left-eye"].delta_z_m, .12)
        self.assertLess(controls["mouth-midline"].delta_z_m, -.07)
        self.assertGreater(controls["chin-bottom"].delta_z_m, .10)
        self.assertAlmostEqual(controls["nose-break"].delta_z_m, 0., delta=.025)
        self.assertLess(abs(controls["left-eye"].delta_y_m), .10)
        self.assertLess(abs(controls["right-eye"].delta_y_m), .10)

    def test_warp_preserves_closed_surface_rear_face_and_local_controls(self):
        face = _anatomical_sphinx_face(scale_x=1., scale_y=1., scale_z=Z_SCALE)
        fit = self.fit()
        warped = deform_sphinx_face(face, fit)
        self.assertEqual(warped.faces, face.faces)
        half = len(face.vertices)//2
        self.assertEqual(warped.vertices[half:], face.vertices[half:])
        report = audit_mesh(warped)
        self.assertEqual((report.boundary_edges, report.nonmanifold_edges), (0, 0))
        self.assertGreater(report.volume_m3, .5)
        max_delta = max(math.dist(a, b) for a, b in zip(face.vertices, warped.vertices))
        self.assertGreater(max_delta, .13)
        self.assertLess(max_delta, .30)
        # The RBF must actually match every named control at a nearby mesh
        # vertex, rather than merely changing unrelated facial coordinates.
        for control in fit.controls:
            index = min(range(half), key=lambda i:
                        (face.vertices[i][1]-control.y_m)**2 +
                        (face.vertices[i][2]-control.z_m)**2)
            old, new = face.vertices[index], warped.vertices[index]
            self.assertAlmostEqual(new[1]-old[1], control.delta_y_m, delta=.012,
                                   msg=control.name)
            self.assertAlmostEqual(new[2]-old[2], control.delta_z_m, delta=.015,
                                   msg=control.name)
        # The top crown and lower neck boundaries are pinned near original.
        crown_index = min(range(half), key=lambda i:
                          face.vertices[i][1]**2 +
                          (face.vertices[i][2]-4.51*Z_SCALE)**2)
        self.assertLess(math.dist(face.vertices[crown_index],
                                  warped.vertices[crown_index]), .012)

    def test_identity_landmarks_leave_original_surface_unchanged(self):
        a = load_landmark_view(REFERENCE)
        zero = prepare_sphinx_face_fit(a, a)
        face = _anatomical_sphinx_face(scale_x=1., scale_y=1., scale_z=Z_SCALE)
        self.assertEqual(deform_sphinx_face(face, zero), face)

    def test_scale_controls_and_invalid_calibration_fail_safely(self):
        ref, cand = load_landmark_view(REFERENCE), load_landmark_view(CANDIDATE)
        fit_big = prepare_sphinx_face_fit(ref, cand, scale_y=2., scale_z=2*Z_SCALE,
                                          max_shift_m=.58)
        fit = self.fit()
        for small, large in zip(fit.controls, fit_big.controls):
            self.assertAlmostEqual(large.delta_y_m, 2*small.delta_y_m, places=5)
            self.assertAlmostEqual(large.delta_z_m, 2*small.delta_z_m, places=5)
        for value in (0., -1., float("nan")):
            with self.subTest(limit=value), self.assertRaises(LandmarkError):
                prepare_sphinx_face_fit(ref, cand, max_shift_m=value)
        with self.assertRaisesRegex(LandmarkError, "excessive movement"):
            prepare_sphinx_face_fit(ref, cand, max_shift_m=.02)
        wrong = LandmarkView("profile-left", ref.size, ref.anchors, ref.landmarks)
        with self.assertRaises(LandmarkError):
            prepare_sphinx_face_fit(wrong, cand)

    def test_monument_plan_supports_opt_in_reference_fitted_geometry(self):
        fit = self.fit()
        study = make_monument_study(kind="sphinx", sphinx_face_fit=fit)
        self.assertEqual(dict(study.parameters)["sphinx_face_fit_sha256"],
                         fit.reference_sha256)
        face = next(part.mesh for part in study.instances if part.name == "sphinx-face")
        self.assertEqual((audit_mesh(face).boundary_edges,
                          audit_mesh(face).nonmanifold_edges), (0, 0))
        standard = make_monument_study(kind="sphinx")
        self.assertNotIn("sphinx_face_fit_sha256", dict(standard.parameters))
        self.assertNotEqual(face.vertices, next(
            part.mesh.vertices for part in standard.instances if part.name == "sphinx-face"))

    def test_cli_requires_both_landmark_files_and_reports_fit_digest(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([
                "plan-monument", "--kind", "sphinx",
                "--sphinx-face-reference", str(REFERENCE),
            ]), 2)
            self.assertEqual(main([
                "plan-monument", "--kind", "pyramid",
                "--sphinx-face-reference", str(REFERENCE),
                "--sphinx-face-candidate", str(CANDIDATE),
            ]), 2)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main([
                "plan-monument", "--kind", "sphinx",
                "--sphinx-face-reference", str(REFERENCE),
                "--sphinx-face-candidate", str(CANDIDATE),
            ])
        self.assertEqual(result, 0)
        self.assertEqual(len(json.loads(output.getvalue())["parameters"][
            "sphinx_face_fit_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
