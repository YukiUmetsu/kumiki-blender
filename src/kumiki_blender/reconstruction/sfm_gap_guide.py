"""Readable, object-agnostic acquisition guides for unregistered SfM views.

Registered COLMAP poses are measured; unregistered view directions are not.
Filename/manifest clues are *labels only*. The guide never invents an azimuth,
object landmark, or a solved bridge, and never modifies the source revision.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _original(staged: dict[str, dict], filename: str | None) -> str | None:
    if filename is None:
        return None
    value = staged.get(filename, {}).get("original_filename")
    return value if isinstance(value, str) and 0 < len(value) <= 512 else filename


def _view_hint(staged: dict[str, dict], filename: str) -> dict:
    """Surface an existing clue without mistaking it for a measured camera pose."""
    record = staged.get(filename, {})
    declared = record.get("view")
    if isinstance(declared, str) and declared.strip().lower() not in ("", "unknown"):
        return {"label": declared[:120], "evidence": "source-manifest / not a measured pose"}
    name = _original(staged, filename) or filename
    words = set(re.findall(r"[a-z]+", name.lower()))
    if words & {"overhead", "top", "aerial", "birdseye"}:
        label = "elevated or overhead view"
    elif words & {"back", "rear", "behind"}:
        label = "rear view" + (" from left" if "left" in words else
                                " from right" if "right" in words else "")
    elif "front" in words:
        label = "frontal view"
    elif words & {"left", "right"}:
        label = ("left" if "left" in words else "right") + " side/profile"
    elif words & {"side", "profile"}:
        label = "side/profile; direction unknown"
    else:
        label = "angle unknown; inspect photograph"
    return {"label": label, "evidence": ("filename hint / unverified" if label !=
            "angle unknown; inspect photograph" else "no reliable view label")}


def annotate_gap_requests(gaps: list[dict], staged: dict[str, dict],
                          anchors: set[str]) -> None:
    """Add immediately actionable original-photo pairs to each evidence gap."""
    for gap in gaps:
        targets = gap["bridge_photo_targets"]
        registered, missing = targets["registered_reference"], targets["missing_reference"]
        anchors_in_orbit = sorted(anchors - {registered} if registered else anchors)
        anchor_examples = [{"staged_filename": name,
                            "original_filename": _original(staged, name)}
                           for name in anchors_in_orbit[:5]]
        missing_examples = [{"staged_filename": name,
                             "original_filename": _original(staged, name),
                             "view_hint": _view_hint(staged, name)}
                            for name in gap["unregistered_filenames"]]
        reference = _original(staged, registered)
        destination = _original(staged, missing)
        pair = ({"registered": {"staged_filename": registered,
                                 "original_filename": reference},
                 "unregistered": {"staged_filename": missing,
                                  "original_filename": destination,
                                  "view_hint": _view_hint(staged, missing)},
                 "existing_orbit_examples": anchor_examples,
                 "other_missing_views": missing_examples,
                 "measured_missing_camera_angle": None,
                 "next_photos_to_find": (
                     f"Find real intermediate photos connecting '{reference}' and "
                     f"'{destination}' to the registered camera group, with successive "
                     "views sharing distinctive *object-surface* details. Seek several "
                     "small viewpoint changes (~10–15 degrees) and substantial subject "
                     "overlap (~60–80%), not repeated crops or background-only matches."
                     if reference else
                     f"Find real, overlapping photos between the registered camera "
                     f"group and '{destination}', beginning with a shared object region."),
                 "confidence": "acquisition hypothesis / not a solved camera direction"})
        gap["photo_request"] = pair


def first_photos_to_find(gaps: list[dict]) -> list[dict]:
    """Put the two named sides of every missing camera gap at the top of JSON."""
    return [{"gap": gap["group"],
             "connect_registered_photo": gap["photo_request"]["registered"]["original_filename"],
             "to_missing_photo": gap["photo_request"]["unregistered"]["original_filename"],
             "missing_view_hint": gap["photo_request"]["unregistered"]["view_hint"],
             "request": gap["photo_request"]["next_photos_to_find"]}
            for gap in gaps]


def _md(value: object) -> str:
    """Safe inline Markdown code label, including hostile source filenames."""
    clean = "".join(ch if ch.isprintable() and ch not in "\r\n" else " " for ch in str(value))
    fence = "`" * (max((len(x) for x in re.findall(r"`+", clean)), default=0) + 1)
    return f"{fence} {clean} {fence}" if "`" in clean else f"`{clean}`"


def markdown_gap_guide(report: dict) -> str:
    """A concise checklist containing original filenames and explicit uncertainty."""
    gaps = report.get("coverage_gaps", [])
    lines = ["# Missing camera angles and bridge-photo requests", "",
             f"Source: {_md(report['source_revision'])}", "",
             "The unregistered photographs have **no measured COLMAP camera angles**. "
             "View labels below come from the source manifest or filename only and "
             "must be verified against the photos. All suggestions are acquisition "
             "hypotheses, not proven matches.", ""]
    for gap in gaps:
        request = gap["photo_request"]
        a, b = request["registered"], request["unregistered"]
        lines.extend([f"## {gap['group']}: {gap['missing_cameras']} missing cameras", "",
                      "**Find intermediate views connecting:**", ""])
        if a["original_filename"]:
            lines.append(f"- Registered reference: {_md(a['original_filename'])} "
                         f"(staged {_md(a['staged_filename'])})")
        else:
            lines.append("- Registered reference: no strong verified neighbor; "
                         "start from the registered orbit")
        lines.extend([f"- Missing reference: {_md(b['original_filename'])} "
                      f"(staged {_md(b['staged_filename'])})",
                      f"- Missing view clue: {_md(b['view_hint']['label'])} "
                      f"({b['view_hint']['evidence']})", "",
                      "Look for **real intermediate photographs** connecting "
                      + (_md(a["original_filename"]) + " and " if a["original_filename"]
                         else "the registered camera group and ")
                      + _md(b["original_filename"])
                      + ". Keep identifiable object-surface features visible across "
                        "successive 10–15° changes, with approximately 60–80% subject "
                        "overlap. Avoid repeated crops and background-only matches.", "",
                      "**Other missing views in this group:**", ""])
        for example in request["other_missing_views"]:
            hint = example["view_hint"]
            lines.append(f"- {_md(example['original_filename'])}: {_md(hint['label'])} "
                         f"({hint['evidence']})")
        lines.extend(["", "**Existing registered orbit examples:** " +
                      (", ".join(_md(v["original_filename"]) for v in
                                  request["existing_orbit_examples"])
                       or "none selected"), "",
                      f"Evidence: {gap['verified_pairs_to_registered']} verified bridge "
                      "pairs; this count does not establish a valid camera alignment.", ""])
    if not gaps:
        lines.append("No missing-view groups were reported.\n")
    lines.extend(["## What to photograph or search for", "",
                  "1. Start with the two *original filenames* in each gap above. "
                  "View them side by side (or use the contact sheet below).",
                  "2. Find a short sequence of actual intermediate viewpoints, "
                  "showing the same identifiable object details across adjacent images.",
                  "3. Prefer original-resolution, uncropped photos with different camera "
                  "positions; avoid AI-generated angles, screenshots of 3D renders, "
                  "repeated photos, and matches only on the scenery.",
                  "4. Add candidates with `prepare-sfm-bridge-photos`, rerun sparse "
                  "matching, and validate 2D-to-3D tracks and camera poses.", ""])
    return "\n".join(lines)


def _contact_sheet(report: dict, sparse: Path, destination: Path) -> dict:
    """Preview original and missing stages per group. Pillow is an optional extra."""
    try:
        from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
    except ImportError:
        return {"status": "pillow-unavailable", "path": None}
    # Bound resource use even for unexpectedly large uploaded photo collections.
    cards = []
    for gap in report.get("coverage_gaps", [])[:24]:
        req = gap["photo_request"]
        ref = req["registered"]
        if ref["staged_filename"]:
            cards.append((gap["group"], "REGISTERED", ref))
        # Show *all* missing view types, not only the best bridge candidate.
        for entry in req["other_missing_views"][:12]:
            cards.append((gap["group"], "MISSING", entry))
        if len(cards) >= 144:
            break
    if not cards:
        return {"status": "no-missing-views", "path": None}
    width, height, columns = 366, 275, 4
    rows = (len(cards) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows), (26, 29, 34))
    draw = ImageDraw.Draw(canvas)
    skipped = []
    for index, (gap, role, photo) in enumerate(cards):
        left, top = (index % columns) * width, (index // columns) * height
        draw.rectangle((left + 2, top + 2, left + width - 2, top + height - 2),
                       fill=(38, 42, 48))
        filename = photo["staged_filename"]
        target = sparse / "images" / filename
        try:
            with Image.open(target) as raw:
                if raw.width * raw.height > 40_000_000:
                    raise ValueError("image exceeds 40MP preview limit")
                raw.draft("RGB", (width * 2, height * 2))
                preview = ImageOps.exif_transpose(raw).convert("RGB")
                preview.thumbnail((width - 16, height - 70))
                canvas.paste(preview, (left + (width - preview.width) // 2,
                                       top + 10 + (height - 70 - preview.height) // 2))
        except (OSError, ValueError, UnidentifiedImageError) as error:
            skipped.append({"filename": filename, "reason": type(error).__name__})
            draw.text((left + 12, top + 90), "Preview unavailable", fill=(230, 185, 95))
        draw.text((left + 10, top + 211), f"{gap} / {role}",
                  fill=(125, 218, 160) if role == "REGISTERED" else (240, 177, 106))
        draw.text((left + 10, top + 232), str(photo["original_filename"])[:48],
                  fill=(240, 240, 240))
        draw.text((left + 10, top + 251), filename[:48], fill=(181, 185, 190))
    canvas.save(destination, format="JPEG", quality=86)
    return {"status": "created", "path": str(destination),
            "cards": len(cards), "unavailable_previews": skipped}


def write_gap_guide(report: dict, revision: Path, output: Path) -> dict:
    """Write optional explainable artifacts only to a new, independent folder."""
    source = revision.resolve()
    destination = output.resolve()
    if (output.is_symlink() or destination.exists() or
            destination == source or source in destination.parents):
        raise ValueError("Guide output must be a new directory outside the source revision")
    sparse = source / "sparse-reconstruction" if (source / "sparse-reconstruction").is_dir() else source
    destination.mkdir(parents=True, exist_ok=False)
    sheet = _contact_sheet(report, sparse, destination / "missing-camera-angles.jpg")
    markdown = markdown_gap_guide(report)
    if sheet["status"] == "created":
        markdown += "\n## Side-by-side camera-gap preview\n\n![Missing-view contact sheet](missing-camera-angles.jpg)\n"
    (destination / "missing-angle-guide.md").write_text(markdown, encoding="utf-8")
    (destination / "missing-angle-guide.json").write_text(
        json.dumps({"first_photos_to_find": report.get("first_photos_to_find", []),
                    "coverage_gaps": report.get("coverage_gaps", []),
                    "source_revision": report["source_revision"]}, indent=2) + "\n",
        encoding="utf-8")
    return {"markdown": str(destination / "missing-angle-guide.md"),
            "json": str(destination / "missing-angle-guide.json"),
            "contact_sheet": sheet}
