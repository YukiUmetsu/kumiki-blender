"""Rank withheld mask proposals for human review with independent evidence.

Automatic mask gates compare a proposal with the current mask, so they must
refuse large changes. That refusal is correct for approval, but it also hides
good proposals when the current mask is the broken one. This module ranks
those withheld proposals for review. It works for any subject: it reads only
registered cameras, feature tracks, held-out evidence and masks.

It never approves or replaces a mask. A person promotes a proposal with
`select-track-masks`, and normal mask, camera and mesh QA still apply.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .dense import _sha256
from .mask_track_guidance import _safe_pinned_image
from .references import ReferenceError


SCHEMA = "mask-proposal-review-queue-v1"
QUEUE_REPORT = "mask-review-queue.json"
# Receipt key in each mask-report image -> proposal kind for select-track-masks.
PROPOSAL_RECEIPTS = {"track-guided": "track_guidance",
                     "component-guided": "component_guidance"}
RECOMMENDED = "review-recommended"
LEAKAGE_UNTESTED = "improved-leakage-untested"
VETOED = "vetoed-by-independent-evidence"
INSUFFICIENT = "insufficient-evidence"
NO_GAIN = "no-cross-view-improvement"
DECISION_ORDER = (RECOMMENDED, LEAKAGE_UNTESTED, INSUFFICIENT, NO_GAIN, VETOED)


@dataclass(frozen=True)
class ReviewPolicy:
    """Evidence limits for a review recommendation. Limits never approve a mask."""

    min_foreground_holdouts: int = 3
    min_background_tracks: int = 3
    min_consensus_tracks: int = 25
    min_peer_cameras: int = 4
    min_peer_agreement: float = .8
    min_agreement_gain: float = .02
    max_new_sky_pixels: int = 0
    max_review_images: int = 24

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            minimum = 0 if field.name == "max_new_sky_pixels" else 1
            if field.type == "int" and (type(value) is not int or value < minimum):
                raise ReferenceError(f"Review policy {field.name} must be an integer of "
                                     f"at least {minimum}; got {value!r}")
        if (type(self.min_peer_agreement) is not float or
                not .5 < self.min_peer_agreement <= 1):
            raise ReferenceError("Review policy min_peer_agreement must be a number in "
                                 f"(0.5, 1]; got {self.min_peer_agreement!r}")
        if (type(self.min_agreement_gain) is not float or
                not 0 <= self.min_agreement_gain <= 1):
            raise ReferenceError("Review policy min_agreement_gain must be a number in "
                                 f"[0, 1]; got {self.min_agreement_gain!r}")

    @classmethod
    def from_json(cls, path: str | Path) -> "ReviewPolicy":
        """Load overrides. Unknown names fail, so a typo cannot be ignored."""
        policy_file = Path(path)
        if policy_file.is_symlink() or not policy_file.is_file():
            raise ReferenceError(f"Review policy file is missing: {policy_file}")
        try:
            values = json.loads(policy_file.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ReferenceError(f"Review policy is not valid JSON: {policy_file}") from error
        known = {field.name for field in fields(cls)}
        if not isinstance(values, dict) or set(values) - known:
            raise ReferenceError("Review policy must be a JSON object using only: " +
                                 ", ".join(sorted(known)))
        return cls(**values)


def withheld_proposals(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return pinned proposals that the automatic gate did not select."""
    withheld = {}
    for kind, receipt_key in PROPOSAL_RECEIPTS.items():
        receipt = entry.get(receipt_key) or {}
        if (not receipt.get("accepted") and receipt.get("proposed_mask") and
                receipt.get("proposed_mask_sha256") not in (None, entry.get("mask_sha256"))):
            withheld[kind] = receipt
    return withheld


def single_view_tracks(observations: list[tuple[int, float, float]]
                       ) -> list[tuple[int, float, float]]:
    """Drop tracks seen more than once in one photo: their location is ambiguous."""
    counts = Counter(point_id for point_id, _, _ in observations)
    return [row for row in observations if counts[row[0]] == 1]


def _label(mask: Any, x: float, y: float) -> bool | None:
    height, width = mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    return bool(mask[min(height-1, round(y)), min(width-1, round(x))])


def collect_track_votes(masks: dict[str, Any],
                        tracks: dict[str, list[tuple[int, float, float]]]
                        ) -> dict[int, dict[str, bool]]:
    """Record each camera's foreground label for each 3D track."""
    votes: dict[int, dict[str, bool]] = {}
    for name, mask in masks.items():
        for point_id, x, y in single_view_tracks(tracks[name]):
            label = _label(mask, x, y)
            if label is not None:
                votes.setdefault(point_id, {})[name] = label
    return votes


def score_cross_view(name: str, kept: Any, proposal: Any,
                     observations: list[tuple[int, float, float]],
                     votes: dict[int, dict[str, bool]],
                     policy: ReviewPolicy) -> dict[str, Any]:
    """Compare both masks with the *other* cameras; a view never votes for itself."""
    consensus_tracks = kept_agrees = proposal_agrees = 0
    repaired = damaged = new_background = 0
    for point_id, x, y in single_view_tracks(observations):
        peers = [label for camera, label in votes.get(point_id, {}).items() if camera != name]
        kept_label, proposal_label = _label(kept, x, y), _label(proposal, x, y)
        if len(peers) < policy.min_peer_cameras or kept_label is None or proposal_label is None:
            continue
        foreground_votes = sum(peers)
        majority = foreground_votes * 2 > len(peers)
        support = (foreground_votes if majority else len(peers) - foreground_votes) / len(peers)
        if support < policy.min_peer_agreement:
            continue
        consensus_tracks += 1
        kept_agrees += kept_label == majority
        proposal_agrees += proposal_label == majority
        repaired += kept_label != majority and proposal_label == majority
        damaged += kept_label == majority and proposal_label != majority
        new_background += proposal_label and not kept_label and not majority
    kept_agreement = kept_agrees / consensus_tracks if consensus_tracks else 0.
    proposal_agreement = proposal_agrees / consensus_tracks if consensus_tracks else 0.
    return {"consensus_tracks": consensus_tracks,
            "kept_agreement": round(kept_agreement, 5),
            "proposal_agreement": round(proposal_agreement, 5),
            "agreement_gain": round(proposal_agreement - kept_agreement, 5),
            "repaired_tracks": repaired, "damaged_tracks": damaged,
            "new_background_tracks": new_background}


def count_veto_evidence(kept: Any, proposal: Any, *,
                        foreground_holdouts: list[dict], background_rows: list[dict],
                        negative_prompts: list[list[float]]) -> dict[str, int]:
    """Count contradictions from evidence that did not create the proposal."""
    def hit(mask: Any, xy: list[float]) -> bool:
        height, width = mask.shape
        return bool(mask[min(height-1, round(xy[1]*(height-1))),
                         min(width-1, round(xy[0]*(width-1)))])

    return {
        "foreground_holdouts": len(foreground_holdouts),
        "background_tracks": len(background_rows),
        "foreground_holdouts_lost": sum(
            hit(kept, row["normalized_xy"]) and not hit(proposal, row["normalized_xy"])
            for row in foreground_holdouts),
        "background_tracks_included": sum(
            hit(proposal, row["normalized_xy"]) for row in background_rows),
        "negative_prompts_included": sum(hit(proposal, xy) for xy in negative_prompts)}


def decide_review(cross_view: dict[str, Any], veto: dict[str, int],
                  new_sky_pixels: int, policy: ReviewPolicy) -> tuple[str, list[str]]:
    """Choose one queue decision. Contradictions always override a metric gain."""
    contradictions = [name for name, count in (
        ("background-track-inside-proposal", veto["background_tracks_included"]),
        ("foreground-holdout-lost", veto["foreground_holdouts_lost"]),
        ("sam2-negative-prompt-inside-proposal", veto["negative_prompts_included"]),
        ("new-blue-sky-foreground", new_sky_pixels - policy.max_new_sky_pixels)) if count > 0]
    if contradictions:
        return VETOED, contradictions
    missing = [name for name, count, minimum in (
        ("too-few-foreground-holdouts", veto["foreground_holdouts"],
         policy.min_foreground_holdouts),
        ("too-few-consensus-tracks", cross_view["consensus_tracks"],
         policy.min_consensus_tracks)) if count < minimum]
    if missing:
        return INSUFFICIENT, missing
    if cross_view["agreement_gain"] < policy.min_agreement_gain:
        return NO_GAIN, ["cross-view-agreement-did-not-improve"]
    # Few background tracks is common and cannot show that the proposal is bad.
    # Keep the proposal visible, but state that leakage was not tested.
    if veto["background_tracks"] < policy.min_background_tracks:
        return LEAKAGE_UNTESTED, ["too-few-background-tracks"]
    return RECOMMENDED, []


def _load_mask(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(path) as mask:
        return np.asarray(mask.convert("L")) >= 128


def _queue_order(item: dict[str, Any]) -> tuple:
    rank = DECISION_ORDER.index(item["decision"])
    return rank, -item["cross_view"]["agreement_gain"], item["filename"], item["proposal"]


def _write_review_image(photo: Any, kept: Any, proposal: Any, path: Path) -> None:
    """Show the photo, the kept mask, the proposal and the difference side by side."""
    import numpy as np
    from PIL import Image

    preview = photo.copy()
    preview.thumbnail((640, 640))

    def small(mask: Any) -> Any:
        binary = Image.fromarray((mask*255).astype("uint8"), "L")
        return np.asarray(binary.resize(preview.size, Image.Resampling.NEAREST)) > 0

    pixels = np.asarray(preview).astype(float)
    old, new = small(kept), small(proposal)

    def tinted(mask: Any) -> Any:
        tile = pixels.copy()
        tile[mask] = tile[mask]*.55 + np.array((238., 58., 35.))*.45
        return Image.fromarray(tile.astype("uint8"), "RGB")

    difference = pixels * .45
    difference[old & ~new] = (255, 40, 40)
    difference[new & ~old] = (0, 230, 255)
    tiles = [preview, tinted(old), tinted(new),
             Image.fromarray(difference.astype("uint8"), "RGB")]
    sheet = Image.new("RGB", (sum(tile.width+4 for tile in tiles), preview.height), "white")
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width + 4
    sheet.save(path, "JPEG", quality=86)


def write_review_queue(sparse: Path, revision: Path, report: dict[str, Any],
                       evidence: dict[str, Any], output: Path,
                       policy: ReviewPolicy) -> dict[str, Any]:
    """Score every withheld proposal in one revision and write the queue."""
    from PIL import Image

    from .mask_track_guidance import _binary_image_tracks, _text_image_tracks
    from .mask_track_resegmentation import _mask_sky_leaks

    model_root = sparse / "undistorted/sparse"
    model = model_root / "images.bin"
    tracks = (_binary_image_tracks(model) if model.is_file() else
              _text_image_tracks(model_root / "images.txt"))
    entries = {entry["filename"]: entry for entry in report["images"]}
    if set(tracks) != set(entries):
        raise ReferenceError("COLMAP track images differ from the mask revision photos; "
                             "rank proposals against the revision's own reconstruction")
    kept_masks = {name: _load_mask(_safe_pinned_image(revision, entry["mask"],
                                                       entry["mask_sha256"]))
                  for name, entry in entries.items()}
    votes = collect_track_votes(kept_masks, tracks)
    output.mkdir(parents=True, exist_ok=True)
    (output / "review").mkdir(exist_ok=True)

    # Decide first; only then spend time on overlays for the items a person reads.
    queue: list[dict[str, Any]] = []
    pending_images: dict[tuple[str, str], Any] = {}
    no_proposal: list[dict[str, Any]] = []
    for name, entry in sorted(entries.items()):
        kept = kept_masks[name]
        proposals = withheld_proposals(entry)
        if not proposals:
            view = score_cross_view(name, kept, kept, tracks[name], votes, policy)
            no_proposal.append({"filename": name, "consensus_tracks": view["consensus_tracks"],
                                "kept_agreement": view["kept_agreement"]})
            continue
        rows = evidence["images"][name]
        with Image.open(sparse / "undistorted/images" / name) as source:
            photo = source.convert("RGB")
        sky_before = _mask_sky_leaks(photo, Image.fromarray((kept*255).astype("uint8"), "L"))
        for kind, receipt in proposals.items():
            path = _safe_pinned_image(revision, receipt["proposed_mask"],
                                      receipt["proposed_mask_sha256"])
            proposal = _load_mask(path)
            if proposal.shape != kept.shape:
                raise ReferenceError(f"Proposal and kept mask sizes differ: {name}")
            cross_view = score_cross_view(name, kept, proposal, tracks[name], votes, policy)
            veto = count_veto_evidence(
                kept, proposal, foreground_holdouts=rows.get("holdout_points", []),
                background_rows=(rows.get("background_seed_points", []) +
                                 rows.get("background_holdout_points", [])),
                negative_prompts=(receipt.get("guide") or {}).get("background_points", []))
            sky_after = _mask_sky_leaks(photo, Image.fromarray((proposal*255).astype("uint8"), "L"))
            decision, reasons = decide_review(cross_view, veto,
                                              max(0, sky_after - sky_before), policy)
            evaluation = receipt.get("evaluation") or receipt
            queue.append({"filename": name, "proposal": kind, "decision": decision,
                          "reasons": reasons, "promote_with": f"{name}:{kind}",
                          "proposed_mask": receipt["proposed_mask"],
                          "proposed_mask_sha256": receipt["proposed_mask_sha256"],
                          "area_ratio": round(int(proposal.sum()) / max(1, int(kept.sum())), 5),
                          "automatic_gate_reasons": evaluation.get("reasons", []),
                          "cross_view": cross_view, "veto_evidence": veto,
                          "sky_pixels_before": sky_before, "sky_pixels_after": sky_after})
            pending_images[(name, kind)] = path
    queue.sort(key=_queue_order)
    no_proposal.sort(key=lambda row: (row["kept_agreement"], row["filename"]))

    for item in queue[:policy.max_review_images]:
        key = (item["filename"], item["proposal"])
        relative = f"review/{item['filename']}.{item['proposal']}.jpg"
        with Image.open(sparse / "undistorted/images" / item["filename"]) as source:
            _write_review_image(source.convert("RGB"), kept_masks[item["filename"]],
                                _load_mask(pending_images[key]), output / relative)
        item["review_image"] = relative

    counts = Counter(item["decision"] for item in queue)
    result = {"schema": SCHEMA, "source_revision": str(sparse), "mask_revision": str(revision),
              "status": "candidate / human-review-required",
              "approved_for_reconstruction": False, "distribution_allowed": False,
              "policy": asdict(policy),
              "summary": {"proposals": len(queue),
                          **{decision: counts[decision] for decision in DECISION_ORDER},
                          "photos_without_proposal": len(no_proposal)},
              "queue": queue, "photos_without_proposal": no_proposal,
              "promote_recommended": [item["promote_with"] for item in queue
                                      if item["decision"] == RECOMMENDED],
              "limitations": ("A recommendation is a review priority, not approval. Peer "
                              "cameras can share one mask error, and sparse tracks cannot "
                              "see textureless surfaces. Inspect each review image: "
                              "photo, kept mask, proposal, difference (red removed, cyan added).")}
    (output / QUEUE_REPORT).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def rank_mask_proposals(reconstruction: str | Path, mask_revision: str | Path,
                        output: str | Path, *,
                        policy: ReviewPolicy | None = None) -> dict[str, Any]:
    """Rank a completed revision's proposals into a new, separate output folder."""
    from .mask_track_resegmentation import TRACK_REVISION_SCHEMAS

    sparse = Path(reconstruction).resolve()
    revision = Path(mask_revision).resolve()
    target = Path(output).resolve()
    if (Path(mask_revision).is_symlink() or not revision.is_dir() or
            Path(output).is_symlink() or target.exists() or
            target.is_relative_to(revision) or target.is_relative_to(sparse)):
        raise ReferenceError("Proposal ranking needs a completed revision and an unused "
                             "output folder outside the revision and reconstruction")
    report_path, evidence_path = revision / "mask-report.json", revision / "track-evidence-report.json"
    for path in (report_path, evidence_path):
        if path.is_symlink() or not path.is_file():
            raise ReferenceError(f"Mask revision is missing {path.name}; "
                                 "run improve-dense-masks to create a complete revision")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("schema") not in TRACK_REVISION_SCHEMAS or
            report.get("source_revision") != str(sparse) or
            not str(report.get("status", "")).startswith("candidate / ")):
        raise ReferenceError("Only a completed track-guided revision of this reconstruction "
                             "can supply proposals")
    if report.get("track_evidence_sha256") != _sha256(evidence_path):
        raise ReferenceError("Track evidence changed after the revision was written")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    return write_review_queue(sparse, revision, report, evidence, target,
                              policy or ReviewPolicy())
