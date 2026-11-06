"""Ranking rules for withheld mask proposals. Ranking never approves a mask."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kumiki_blender.reconstruction.mask_proposal_review import (
    INSUFFICIENT, LEAKAGE_UNTESTED, NO_GAIN, RECOMMENDED, VETOED, ReviewPolicy,
    collect_track_votes, count_veto_evidence, decide_review, score_cross_view,
    single_view_tracks, withheld_proposals)
from kumiki_blender.reconstruction.references import ReferenceError


def veto_counts(**overrides):
    return {"foreground_holdouts": 5, "background_tracks": 5,
            "foreground_holdouts_lost": 0, "background_tracks_included": 0,
            "negative_prompts_included": 0, **overrides}


def view(**overrides):
    return {"consensus_tracks": 100, "agreement_gain": .1, **overrides}


class ReviewDecisionTests(unittest.TestCase):
    def test_strong_evidence_is_recommended_only_for_review(self):
        self.assertEqual(decide_review(view(), veto_counts(), 0, ReviewPolicy()),
                         (RECOMMENDED, []))

    def test_any_contradiction_overrides_a_large_gain(self):
        for field, reason in (("background_tracks_included", "background-track-inside-proposal"),
                              ("foreground_holdouts_lost", "foreground-holdout-lost"),
                              ("negative_prompts_included", "sam2-negative-prompt-inside-proposal")):
            decision, reasons = decide_review(view(agreement_gain=.9), veto_counts(**{field: 1}),
                                              0, ReviewPolicy())
            self.assertEqual(decision, VETOED)
            self.assertIn(reason, reasons)

    def test_new_sky_vetoes_unless_policy_allows_edge_aliasing(self):
        self.assertEqual(decide_review(view(), veto_counts(), 1, ReviewPolicy())[0], VETOED)
        self.assertEqual(decide_review(view(), veto_counts(), 1,
                                       ReviewPolicy(max_new_sky_pixels=1))[0], RECOMMENDED)

    def test_missing_foreground_or_track_evidence_cannot_be_recommended(self):
        policy = ReviewPolicy()
        self.assertEqual(decide_review(view(), veto_counts(foreground_holdouts=2), 0, policy)[0],
                         INSUFFICIENT)
        self.assertEqual(decide_review(view(consensus_tracks=24), veto_counts(), 0, policy)[0],
                         INSUFFICIENT)

    def test_missing_background_evidence_is_reported_as_untested(self):
        self.assertEqual(decide_review(view(), veto_counts(background_tracks=2), 0, ReviewPolicy()),
                         (LEAKAGE_UNTESTED, ["too-few-background-tracks"]))

    def test_no_gain_is_not_promoted_by_missing_background_evidence(self):
        self.assertEqual(decide_review(view(agreement_gain=.01), veto_counts(background_tracks=0),
                                       0, ReviewPolicy())[0], NO_GAIN)


class CrossViewScoreTests(unittest.TestCase):
    def setUp(self):
        self.policy = ReviewPolicy(min_peer_cameras=3, min_peer_agreement=.75)
        self.empty = np.zeros((10, 10), dtype=bool)
        self.full = np.ones((10, 10), dtype=bool)

    def test_target_camera_cannot_vote_for_itself(self):
        votes = {7: {"target": True, "a": False, "b": False, "c": False}}
        score = score_cross_view("target", self.full, self.empty, [(7, 5., 5.)], votes, self.policy)
        self.assertEqual(score["consensus_tracks"], 1)
        self.assertEqual(score["repaired_tracks"], 1)
        self.assertEqual(score["agreement_gain"], 1.)

    def test_added_foreground_against_peer_background_counts_as_leak(self):
        votes = {7: {"a": False, "b": False, "c": False}}
        score = score_cross_view("target", self.empty, self.full, [(7, 5., 5.)], votes, self.policy)
        self.assertEqual(score["damaged_tracks"], 1)
        self.assertEqual(score["new_background_tracks"], 1)

    def test_split_peers_or_too_few_peers_give_no_consensus(self):
        split = {7: {"a": True, "b": False, "c": True, "d": False}}
        few = {7: {"a": True, "b": True}}
        for votes in (split, few):
            score = score_cross_view("target", self.empty, self.full, [(7, 5., 5.)],
                                     votes, self.policy)
            self.assertEqual(score["consensus_tracks"], 0)
            self.assertEqual(score["agreement_gain"], 0.)

    def test_duplicate_or_outside_observations_supply_no_evidence(self):
        duplicate = [(7, 2., 2.), (7, 8., 8.), (8, 50., 50.)]
        self.assertEqual(single_view_tracks(duplicate), [(8, 50., 50.)])
        votes = collect_track_votes({"a": self.full}, {"a": duplicate})
        self.assertEqual(votes, {})


class VetoEvidenceTests(unittest.TestCase):
    def test_counts_contradictions_from_independent_evidence(self):
        kept = np.zeros((11, 11), dtype=bool)
        kept[5, 5] = True
        proposal = np.zeros((11, 11), dtype=bool)
        proposal[0, 0] = True
        result = count_veto_evidence(kept, proposal,
                                     foreground_holdouts=[{"normalized_xy": [.5, .5]}],
                                     background_rows=[{"normalized_xy": [0, 0]}],
                                     negative_prompts=[[0, 0], [1, 1]])
        self.assertEqual(result, {"foreground_holdouts": 1, "background_tracks": 1,
                                  "foreground_holdouts_lost": 1,
                                  "background_tracks_included": 1,
                                  "negative_prompts_included": 1})


class WithheldProposalTests(unittest.TestCase):
    def test_lists_only_unselected_pinned_proposals(self):
        pinned = {"accepted": False, "proposed_mask": "candidates/a", "proposed_mask_sha256": "x"}
        self.assertEqual(set(withheld_proposals({"mask_sha256": "kept", "track_guidance": pinned})),
                         {"track-guided"})
        self.assertEqual(withheld_proposals({"mask_sha256": "kept", "track_guidance":
                                             {**pinned, "accepted": True}}), {})
        self.assertEqual(withheld_proposals({"mask_sha256": "x", "component_guidance": pinned}), {})
        self.assertEqual(withheld_proposals({"mask_sha256": "kept",
                                             "track_guidance": {"accepted": False}}), {})


class ReviewPolicyTests(unittest.TestCase):
    def test_invalid_limits_name_the_setting(self):
        for values, field in (({"min_consensus_tracks": 0}, "min_consensus_tracks"),
                              ({"min_peer_agreement": .5}, "min_peer_agreement"),
                              ({"min_agreement_gain": -.1}, "min_agreement_gain"),
                              ({"max_new_sky_pixels": -1}, "max_new_sky_pixels")):
            with self.assertRaisesRegex(ReferenceError, field):
                ReviewPolicy(**values)

    def test_json_overrides_defaults_and_rejects_typos(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            path.write_text(json.dumps({"min_background_tracks": 1}))
            self.assertEqual(ReviewPolicy.from_json(path).min_background_tracks, 1)
            path.write_text(json.dumps({"min_background_track": 1}))
            with self.assertRaisesRegex(ReferenceError, "using only"):
                ReviewPolicy.from_json(path)
            with self.assertRaisesRegex(ReferenceError, "missing"):
                ReviewPolicy.from_json(Path(folder) / "absent.json")


if __name__ == "__main__":
    unittest.main()
