"""Generate a self-contained, offline-first statue face annotation workbench.

The HTML embeds bounded JPEG *previews*, not source file paths. It remains a
private report artifact and is never stored with the model repository.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from typing import Sequence

from .schema import AnnotationSet, LANDMARK_IDS, Photo
from .workbench_template import TEMPLATE


def build_annotation_workbench(photos: Sequence[Photo],
                               annotations: AnnotationSet | None = None) -> str:
    """Return standalone HTML whose JSON export matches ``load_annotations``.

    Up to 64 private photos may be edited in one document. The preview resize
    is intentionally separate from source dimensions: click coordinates are
    remapped into the original pixel coordinate system in the browser.
    """
    if not 1 <= len(photos) <= 64:
        raise ValueError("Annotation workbench requires 1–64 photos")
    try:
        from PIL import Image
    except ImportError as error:
        raise ValueError("Annotation workbench needs Pillow (install .[qa])") from error

    images: list[dict[str, object]] = []
    preview_size = 0
    for photo in photos:
        view = annotations.images.get(photo.image_id) if annotations else None
        # Bind the embedded pixels to the exact image that intake audited.
        # Read through the same file handle to avoid a path-swap between hash
        # verification and preview decoding.
        with photo.path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != photo.sha256:
                raise ValueError(f"Photo changed since intake: {photo.image_id}")
            source.seek(0)
            with Image.open(source) as original:
                if original.size != (photo.width, photo.height):
                    raise ValueError(f"Photo dimensions changed since intake: {photo.image_id}")
                preview = original.convert("RGB")
                preview.thumbnail((1200, 1400), Image.Resampling.LANCZOS)
                stream = io.BytesIO()
                preview.save(stream, format="JPEG", quality=82, optimize=True)
                data = stream.getvalue()
        preview_size += len(data)
        if preview_size > 40 * 1024 * 1024:
            raise ValueError("Annotation previews exceed 40 MiB; split the photo set")
        images.append({
            "image_id": photo.image_id,
            "width": photo.width,
            "height": photo.height,
            "sha256": photo.sha256,
            "view": view.view if view else photo.view,
            "landmarks": [{"landmark_id": item.landmark_id,
                           "xy_px": list(item.xy_px) if item.xy_px is not None else None,
                           "origin": item.origin, "visibility": item.visibility}
                          for item in view.landmarks] if view else [],
            "outline_px": [list(point) for point in view.outline_px] if view and view.outline_px else None,
            "preview": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
        })

    priority = ("chin-tip", "jaw-left", "jaw-right", "nose-tip", "nose-bridge",
                "eye-inner-left", "eye-inner-right", "eye-outer-left", "eye-outer-right",
                "mouth-left", "mouth-right", "hairline-center", "head-top", "neck-base")
    ordered = [name for name in priority if name in LANDMARK_IDS]
    ordered.extend(sorted(LANDMARK_IDS.difference(ordered)))
    payload = json.dumps({"schema_version": 1, "images": images, "landmark_ids": ordered},
                         ensure_ascii=True, separators=(",", ":"))
    # The JSON is embedded in an HTML script data island. Escape potential
    # closing tags even if future input fields become less restrictive.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return TEMPLATE.replace("__FACE_PAYLOAD__", payload)
