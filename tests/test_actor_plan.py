"""Actor pack and profile validation. No Blender."""

from __future__ import annotations

from pathlib import Path
import unittest

from kumiki_blender.actor_plan import ActorPlanError, parse_actor

PACK = Path("/tmp/pack")
ASSET = {"kind": "image", "uri": "layers/a.png", "width": 10, "height": 20}
CHARACTER = {"characterId": "c", "revision": "r1", "assets": {"a": ASSET, "b": {**ASSET, "uri": "layers/b.png"}},
             "parts": [{"id": "root", "asset": "a", "pivot": {"x": 1, "y": 2}, "z": 0},
                       {"id": "arm", "parent": "root", "asset": "a", "variants": {"up": "b"}, "z": 1}]}
PROFILE = {"schema_version": 1, "height_m": .25}


class ParseActor(unittest.TestCase):
    def test_minimal(self):
        plan = parse_actor(PACK, CHARACTER, PROFILE)
        self.assertEqual(plan.by_id["arm"].parent, "root")
        self.assertEqual(plan.logical_sizes["a"], (10, 20))

    def test_variant_lookup_matches_kumiki(self):
        plan = parse_actor(PACK, CHARACTER, PROFILE)
        arm = plan.by_id["arm"]
        self.assertEqual(plan.asset_of(arm, "up"), "b")
        self.assertEqual(plan.asset_of(arm, "missing"), "a")
        self.assertEqual(plan.asset_of(arm, None), "a")

    def test_rejects_unknown_profile_field(self):
        with self.assertRaisesRegex(ActorPlanError, "unknown fields"):
            parse_actor(PACK, CHARACTER, {**PROFILE, "scale": 2})

    def test_rejects_path_leaving_the_pack(self):
        bad = {**CHARACTER, "assets": {"a": {**ASSET, "uri": "../../etc/x.png"}, "b": ASSET}}
        with self.assertRaisesRegex(ActorPlanError, "leaves the pack"):
            parse_actor(PACK, bad, PROFILE)

    def test_rejects_missing_parent_and_asset(self):
        with self.assertRaisesRegex(ActorPlanError, "unknown parent"):
            parse_actor(PACK, {**CHARACTER, "parts": [{"id": "x", "parent": "nope", "asset": "a"}]}, PROFILE)
        with self.assertRaisesRegex(ActorPlanError, "needs an image asset"):
            parse_actor(PACK, {**CHARACTER, "parts": [{"id": "x", "asset": "nope"}]}, PROFILE)

    def test_rejects_effector_on_unknown_part(self):
        with self.assertRaisesRegex(ActorPlanError, "unknown part"):
            parse_actor(PACK, CHARACTER, {**PROFILE, "effectors": {"hand": {"part": "nope"}}})

    def test_ground_parts_must_exist(self):
        self.assertEqual(parse_actor(PACK, CHARACTER, {**PROFILE, "ground_parts": ["arm"]}).ground_parts, ("arm",))
        with self.assertRaisesRegex(ActorPlanError, "ground_parts"):
            parse_actor(PACK, CHARACTER, {**PROFILE, "ground_parts": ["nope"]})

    def test_rejects_bad_height(self):
        with self.assertRaisesRegex(ActorPlanError, "height_m"):
            parse_actor(PACK, CHARACTER, {**PROFILE, "height_m": 40})


if __name__ == "__main__":
    unittest.main()
