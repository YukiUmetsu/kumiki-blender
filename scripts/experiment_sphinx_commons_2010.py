#!/usr/bin/env python3
"""Reproducible, photo-derived Great Sphinx SfM experiment (no proxy mesh).

Reconstruct cameras and sparse points from five verified Commons photos or a
local multi-view photo folder, preserving source hashes and diagnostics.
A point-cloud GLB is NOT a watertight or photo-faithful monument mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Wknight94, 6 February 2010, Canon EOS 450D; CC BY-SA 3.0 or GFDL.
# Each page has the original file, attribution, licensing and camera metadata.
SOURCES = (
    "Great_Sphinx_2010.jpg",
    "Great_Sphinx_2010_2.jpg",
    "Great_Sphinx_2010_4.jpg",
    "Great_Sphinx_2010_5.jpg",
    "Great_Sphinx_2010_rear.jpg",
)
AUTHOR = "Wknight94"
LICENSE = "https://creativecommons.org/licenses/by-sa/3.0/"


def commons_page(filename: str) -> str:
    return "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(filename)


def original_url(filename: str) -> str:
    """Commons originals use the MD5-hashed MediaWiki file path."""
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return ("https://upload.wikimedia.org/wikipedia/commons/"
            + digest[0] + "/" + digest[:2] + "/" + urllib.parse.quote(filename))


def download_photo(filename: str, destination: Path, *, offline_dir: Path | None = None) -> dict:
    from PIL import Image

    if offline_dir:
        source = offline_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing offline Commons photograph: {source}")
        shutil.copyfile(source, destination)
    else:
        url = original_url(filename)
        request = urllib.request.Request(url, headers={
            "User-Agent": "KumikiBlenderPhotogrammetryExperiment/0.1 (https://github.com/YukiUmetsu/kumiki-blender)",
            "Accept": "image/jpeg,image/*",
        })
        error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {url}")
                    # Limit download size; no untrusted, unlimited downloads.
                    payload = response.read(20 * 1024 * 1024 + 1)
                if not payload or len(payload) > 20 * 1024 * 1024:
                    raise ValueError(f"Invalid original size: {filename}")
                destination.write_bytes(payload)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    # Wikimedia explicitly requires respecting 429 and Retry-After.
                    # Do not retry, switch hosts, or rotate user agents to evade it.
                    delay = exc.headers.get("Retry-After") if exc.headers else None
                    hint = f" Retry-After: {delay}." if delay else ""
                    raise RuntimeError(
                        f"Wikimedia rate-limited the photo download (HTTP 429).{hint} "
                        "Use --photos-dir /path/to/existing-sphinx-photos to work offline, "
                        "or --offline-photos /path/to/manually-downloaded-commons-originals."
                    ) from exc
                error = exc
                if attempt == 2 or exc.code < 500:
                    raise RuntimeError(f"Could not download {commons_page(filename)}: {error}") from error
                time.sleep(attempt + 1)
            except (OSError, ValueError) as exc:
                error = exc
                if attempt == 2:
                    raise RuntimeError(f"Could not download {commons_page(filename)}: {error}") from error
                time.sleep(attempt + 1)
    with Image.open(destination) as photo:
        photo.verify()
    with Image.open(destination) as photo:
        width, height = photo.size
    if min(width, height) < 640:
        raise ValueError(f"Original photograph too small for SfM: {filename}")
    return {"file": filename, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "width": width, "height": height,
            "original_url": original_url(filename), "source_page": commons_page(filename),
            "author": AUTHOR, "license": LICENSE}


def prepare_sources(target: Path, offline_dir: Path | None) -> tuple[Path, list[dict]]:
    images = target / "input" / "photos"
    images.mkdir(parents=True)
    metadata = []
    entries = []
    for index, filename in enumerate(SOURCES, 1):
        info = download_photo(filename, images / filename, offline_dir=offline_dir)
        metadata.append(info)
        entries.append({"id": f"wknight94-{index:02d}", "file": f"photos/{filename}",
                        "source_url": commons_page(filename), "view": "unknown",
                        "regions": ["object"], "expected_sha256": info["sha256"]})
    manifest = {
        "schema_version": 1, "asset_id": "great-sphinx-2010-wknight94",
        "subject": "Great Sphinx of Giza, 2010 Wikimedia Commons survey experiment",
        "required_views": ["unknown"], "required_regions": ["object"], "images": entries,
    }
    path = target / "input" / "reference-set.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (target / "source-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                  encoding="utf-8")
    credits = [
        "# Source photographs and attribution",
        "",
        "Original photographs: Wknight94, 6 February 2010, Wikimedia Commons.",
        "License: Creative Commons Attribution-ShareAlike 3.0 Unported.",
        LICENSE,
        "",
        "The five original photographs are NOT bundled with this report.",
        "If redistributing a derivative, review the attribution and ShareAlike terms.",
        "Camera registration and point-cloud export are experimental transformations.",
        "",
    ]
    credits += [f"- [{item['file']}]({item['source_page']})" for item in metadata]
    (target / "CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    return path, metadata


def prepare_local_sources(target: Path, photos_dir: Path) -> tuple[Path, list[dict]]:
    """Stage an existing multi-view photo folder without making network requests.

    The photos' public provenance and reuse rights are unknown. Do not claim
    Wikimedia authorship, licensing, or fidelity for these local inputs.
    """
    from PIL import Image

    if not photos_dir.is_dir():
        raise FileNotFoundError(f"Photo directory does not exist: {photos_dir}")
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    originals = sorted(
        (p for p in photos_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: p.name.casefold(),
    )
    if not 3 <= len(originals) <= 256:
        raise ValueError(f"Need 3–256 local photos; found {len(originals)} in {photos_dir}")
    if len({p.name.casefold() for p in originals}) != len(originals):
        raise ValueError("Local photos have case-insensitive filename collisions")
    images = target / "input" / "photos"
    images.mkdir(parents=True)
    metadata, entries = [], []
    for index, source in enumerate(originals, 1):
        if source.is_symlink() or source.stat().st_size > 50 * 1024 * 1024:
            raise ValueError(f"Refusing symlink or oversized source: {source.name}")
        destination = images / source.name
        shutil.copyfile(source, destination)
        with Image.open(destination) as photo:
            photo.verify()
        with Image.open(destination) as photo:
            width, height = photo.size
        if min(width, height) < 640 or width * height > 40_000_000:
            raise ValueError(f"Photo dimensions unsuitable for SfM: {source.name}")
        sha = hashlib.sha256(destination.read_bytes()).hexdigest()
        metadata.append({"file": source.name, "sha256": sha, "width": width,
                         "height": height, "source_kind": "user-upload",
                         "author": "unverified", "license": "unverified"})
        entries.append({"id": f"local-{index:04d}", "file": f"photos/{source.name}",
                        "source_kind": "user-upload", "view": "unknown",
                        "regions": ["object"], "expected_sha256": sha})
    manifest = {
        "schema_version": 1, "asset_id": "great-sphinx-local-multiview",
        "subject": "Great Sphinx of Giza: user-provided local photo experiment",
        "required_views": ["unknown"], "required_regions": ["object"], "images": entries,
    }
    path = target / "input" / "reference-set.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (target / "source-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                  encoding="utf-8")
    (target / "CREDITS.md").write_text(
        "# Local photographs\n\n"
        "Photographs supplied by the user. Public source, author, and reuse "
        "rights have **not** been verified. Do not redistribute these images "
        "or derivatives without permission.\n",
        encoding="utf-8",
    )
    return path, metadata


def export_sparse_point_cloud(sparse: Path, model: str, target: Path) -> dict:
    """Export observed SfM points only; never manufacture an inferred surface."""
    import numpy as np
    import trimesh

    points_file = sparse / "text-models" / model / "points3D.txt"
    if not points_file.is_file():
        return {"status": "unavailable", "reason": "No COLMAP points3D.txt"}
    vertices = []
    colors = []
    for line in points_file.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            xyz = [float(v) for v in fields[1:4]]
            rgb = [int(v) for v in fields[4:7]]
        except ValueError:
            continue
        if not np.isfinite(xyz).all() or any(not 0 <= v <= 255 for v in rgb):
            continue
        vertices.append(xyz)
        colors.append(rgb + [255])
    if not vertices:
        return {"status": "unavailable", "reason": "No finite triangulated points"}
    cloud = trimesh.PointCloud(np.asarray(vertices), colors=np.asarray(colors, dtype=np.uint8))
    cloud.export(str(target / "sparse-points.ply"))
    cloud.export(str(target / "sparse-points.glb"))
    return {"status": "sparse-point-cloud-only", "points": len(vertices),
            "ply": "sparse-points.ply", "glb": "sparse-points.glb",
            "warning": "The GLB contains points, NOT a triangulated surface mesh."}


def diagnose(revision: Path, target: Path) -> dict:
    script = Path(__file__).resolve().parent / "diagnose_splat_registration.py"
    spec = importlib.util.spec_from_file_location("diagnose_splat_registration", script)
    if spec is None or spec.loader is None:
        return {"status": "unavailable", "reason": "Diagnostic script unavailable"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The diagnostic expects <revision>/sparse-reconstruction/...
    graph = module.analyze(revision, 15)
    (target / "registration-graph.json").write_text(json.dumps(graph, indent=2) + "\n",
                                                    encoding="utf-8")
    try:
        module.make_contact_sheet(revision, graph, target / "registration-contact-sheet.jpg")
    except (OSError, ValueError) as error:
        graph["contact_sheet_error"] = str(error)
    return {"status": "analyzed", "registered": graph["registered"],
            "strong_pairs": graph["strong_pair_count"],
            "component_sizes": graph["connected_component_sizes"],
            "models": graph.get("sparse_models", [])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--offline-photos", type=Path,
                         help="Use five manually downloaded Commons originals with exact filenames")
    sources.add_argument("--photos-dir", type=Path,
                         help="Use ALL local JPG, PNG and WebP photos without downloading anything")
    parser.add_argument("--max-image-size", type=int, default=2400)
    parser.add_argument("--profile", choices=("standard", "high-recall"), default="high-recall")
    args = parser.parse_args(argv)
    if args.output.is_symlink() or args.output.exists():
        parser.error("Output already exists: choose a new experiment revision")
    target = args.output.resolve()
    target.mkdir(parents=True)
    summary = {"experiment": "local-sphinx-multiview" if args.photos_dir else "wknight94-2010-sphinx",
               "source_photos": 0, "status": "in-progress", "mesh_generated": False,
               "limitations": "Sparse photogrammetry is not a faithful Sphinx surface mesh."}

    def checkpoint() -> None:
        (target / "result.json").write_text(json.dumps(summary, indent=2) + "\n",
                                            encoding="utf-8")
    checkpoint()
    try:
        if args.photos_dir:
            manifest, metadata = prepare_local_sources(target, args.photos_dir)
        else:
            manifest, metadata = prepare_sources(target, args.offline_photos)
        summary["source_photos"] = len(metadata)
        summary["photos"] = metadata
        checkpoint()
        from kumiki_blender.reconstruction.references import load_reference_set
        from kumiki_blender.reconstruction.colmap import build_sparse_reconstruction
        references = load_reference_set(manifest)
        sparse = target / "sparse-reconstruction"
        report = build_sparse_reconstruction(
            references, sparse, engine="pycolmap", timeout_seconds=1800,
            undistort=True, max_image_size=args.max_image_size,
            registration_profile=args.profile,
        )
        summary["camera_reconstruction"] = report
        summary["status"] = report["status"]
        checkpoint()
        if (sparse / "database.db").is_file():
            try:
                summary["registration_diagnostics"] = diagnose(target, target)
            except (OSError, ValueError, KeyError) as error:
                summary["diagnostic_error"] = str(error)
        models = report.get("models", [])
        if models:
            best = max(models, key=lambda item: item["registered_images"])
            summary["best_model"] = best
            summary["point_cloud"] = export_sparse_point_cloud(sparse, best["model"], target)
        else:
            summary["point_cloud"] = {"status": "unavailable", "reason": "No sparse models"}
        # Sparse points alone cannot establish a usable surface, even when
        # more than eight input photos were provided. Never claim a mesh.
        summary["surface_mesh"] = "not-generated / insufficient-validated-dense-geometry"
        checkpoint()
        print(json.dumps({"status": summary["status"], "best_model": summary.get("best_model"),
                          "point_cloud": summary["point_cloud"],
                          "registration_diagnostics": summary.get("registration_diagnostics")},
                         indent=2))
        return 0
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        checkpoint()
        print(summary["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
