"""Private offline face annotation workbench for multi-angle sculpture photos."""

import base64
import json
import re
import shutil
import subprocess

import pytest
from PIL import Image

from kumiki_blender.face_analysis.__main__ import main
from kumiki_blender.face_analysis.inputs import load_photo_set
from kumiki_blender.face_analysis.schema import Annotation, AnnotationSet, ViewAnnotations
from kumiki_blender.face_analysis.workbench import build_annotation_workbench


def _image(path, color=(140, 150, 160)):
    Image.new("RGB", (180, 220), color).save(path)
    return path


def _payload(html):
    match = re.search(r'<script type="application/json" id="face-data">(.*?)</script>',
                      html, re.DOTALL)
    assert match, "Self-contained data payload missing"
    return json.loads(match.group(1))


def test_editor_embeds_private_previews_and_reviewed_annotations_without_paths(tmp_path):
    photos = load_photo_set([_image(tmp_path / "face1.png"), _image(tmp_path / "face2.png")])
    annotations = AnnotationSet({"face1": ViewAnnotations("face1", "front", (
        Annotation("face1", "chin-tip", (70, 160), "manual", "visible"),
    ), ((20, 10), (170, 10), (170, 200), (20, 200)))})
    html = build_annotation_workbench(photos, annotations)
    payload = _payload(html)
    assert len(payload["images"]) == 2
    assert payload["images"][0]["image_id"] == "face1"
    assert payload["images"][0]["width"] == 180
    assert payload["images"][0]["height"] == 220
    assert payload["images"][0]["sha256"] == photos[0].sha256
    assert payload["images"][0]["view"] == "front"
    assert payload["images"][0]["landmarks"][0]["landmark_id"] == "chin-tip"
    assert payload["images"][0]["outline_px"] == [[20, 10], [170, 10], [170, 200], [20, 200]]
    assert payload["images"][1]["view"] == "unknown"
    assert all(entry["preview"].startswith("data:image/jpeg;base64,") for entry in payload["images"])
    assert all(base64.b64decode(entry["preview"].split(",", 1)[1]).startswith(b"\xff\xd8")
               for entry in payload["images"])
    assert str(tmp_path) not in html
    assert "Export JSON" in html and "Undo" in html
    assert "connect-src 'none'" in html
    assert 'id="point-visibility"' in html


def test_inspect_with_workbench_creates_immutable_local_html(tmp_path):
    source = tmp_path / "photos"
    source.mkdir()
    _image(source / "face1.png")
    output = tmp_path / "revision-r01"
    assert main(["inspect", "--images-dir", str(source), "--workbench",
                 "--output", str(output)]) == 0
    html = output / "annotation-workbench.html"
    assert html.is_file()
    assert _payload(html.read_text())["images"][0]["image_id"] == "face1"
    manifest = json.loads((output / "manifest.json").read_text())
    assert "annotation-workbench.html" in manifest["files"]
    assert manifest["contains_private_photo_previews"] is True
    assert manifest["three_d_fidelity_verified"] is False
    assert not (source / "annotation-workbench.html").exists()
    assert main(["inspect", "--images-dir", str(source), "--workbench",
                 "--output", str(output)]) != 0


def test_workbench_requires_small_bounded_set_of_photos(tmp_path):
    photos = load_photo_set([_image(tmp_path / "face.png")])
    many = photos * 65
    with pytest.raises(ValueError, match="1–64"):
        build_annotation_workbench(many)


def test_embedded_editor_script_has_valid_js_syntax(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node.js not installed")
    photos = load_photo_set([_image(tmp_path / "face.png")])
    html = build_annotation_workbench(photos)
    matches = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    assert len(matches) == 1
    check = subprocess.run(["node", "--check", "-"], input=matches[0], text=True,
                           capture_output=True, timeout=10, check=False)
    assert check.returncode == 0, check.stderr
