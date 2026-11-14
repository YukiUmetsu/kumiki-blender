"""Semantic skin-weight audit invariants independent of Blender."""
from __future__ import annotations

import math
import unittest

from kumiki_blender.character_build.weight_zone_metrics import audit_weight_zone


class WeightZoneMetricsTest(unittest.TestCase):
    def test_allowed_wing_bones_with_foreign_influences_are_rejected(self):
        data = [
            (0, 1.0, {"upper_wing.L": .98, "spine": .02}),
            (1, 1.0, {"upper_wing.L": .7, "fore_wing.L": .2, "head": .1}),
        ]
        report = audit_weight_zone(data, {"upper_wing.L", "fore_wing.L"},
                                   max_outside_weight=.03)
        self.assertEqual(report["selected_vertices"], 2)
        self.assertEqual(report["violating_vertices"], 1)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["worst_examples"][0]["outside_bones"], ["head"])
        self.assertAlmostEqual(report["max_outside_weight_observed"], .1)

    def test_foreign_weights_sum_across_bones(self):
        report = audit_weight_zone(
            [(7, .8, {"wing": .96, "head": .025, "spine": .015})],
            {"wing"}, max_outside_weight=.03)
        self.assertEqual(report["violating_vertices"], 1)
        self.assertAlmostEqual(report["max_outside_weight_observed"], .04)

    def test_no_region_vertices_does_not_pass(self):
        report = audit_weight_zone([(0, .49, {"wing": 1.0})],
                                   {"wing"}, min_vertices=1)
        self.assertEqual(report["selected_vertices"], 0)
        self.assertEqual(report["status"], "fail")

    def test_membership_and_threshold_are_explicit(self):
        report = audit_weight_zone(
            [(0, .5, {"tail": .99, "spine": .01}),
             (1, .4, {"tail": .3, "head": .7})],
            {"tail"}, max_outside_weight=.02)
        self.assertEqual(report["selected_vertices"], 1)
        self.assertEqual(report["status"], "pass")

    def test_invalid_nan_duplicate_indices_and_malformed_weights_fail(self):
        rows = [
            [(0, float("nan"), {"wing": 1.0})],
            [(0, 1.0, {"wing": math.inf})],
            [(0, 1.0, {"wing": 1.0}), (0, 1.0, {"wing": 1.0})],
            [(-1, 1.0, {"wing": 1.0})],
        ]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                audit_weight_zone(row, {"wing"})

    def test_negative_threshold_or_empty_allowed_bones_rejected(self):
        with self.assertRaises(ValueError):
            audit_weight_zone([], set())
        with self.assertRaises(ValueError):
            audit_weight_zone([], {"wing"}, max_outside_weight=-.1)
        with self.assertRaises(ValueError):
            audit_weight_zone([], {"wing"}, min_vertices=0)
        with self.assertRaises(ValueError):
            audit_weight_zone([], {"wing"}, membership_threshold=0)


if __name__ == "__main__":
    unittest.main()
