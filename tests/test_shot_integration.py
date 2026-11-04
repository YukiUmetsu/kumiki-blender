"""Real Blender renders. Skipped where Blender is not installed."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
BLENDER = shutil.which(os.environ.get("KUMIKI_BLENDER", "blender")) or \
    ("/Applications/Blender.app/Contents/MacOS/Blender" if Path("/Applications/Blender.app").exists() else None)


def png(path: Path, width: int, height: int, fill) -> None:
    """Write RGBA from fill(x, y) -> (r, g, b, a) with no image dependency."""
    rows = b"".join(b"\0" + b"".join(bytes(fill(x, y)) for x in range(width)) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def repeat(value, count):
    return [value for _ in range(count)]


@unittest.skipUnless(BLENDER and shutil.which("ffmpeg"), "needs Blender and FFmpeg")
class BuildShot(unittest.TestCase):
    FRAMES = 3

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        pack = self.root / "pack"
        (pack / "layers").mkdir(parents=True)
        png(pack / "layers/body.png", 40, 60, lambda x, y: (120, 90, 60, 255) if 6 < x < 34 else (0, 0, 0, 0))
        png(pack / "layers/arm.png", 60, 12, lambda x, y: (200, 180, 120, 255) if 2 <= y <= 9 else (0, 0, 0, 0))
        asset = {"kind": "image", "mediaType": "image/png"}
        (pack / "character.json").write_text(json.dumps({
            "characterId": "generic", "revision": "r1",
            "assets": {"body": {**asset, "uri": "layers/body.png", "width": 40, "height": 60},
                       "arm": {**asset, "uri": "layers/arm.png", "width": 60, "height": 12}},
            "parts": [{"id": "root", "asset": "body", "pivot": {"x": 20, "y": 30}, "z": 0},
                      {"id": "arm", "parent": "root", "asset": "arm", "pivot": {"x": 2, "y": 6}, "rest": {"x": 18, "y": 10, "scaleX": 1.4, "scaleY": 1.4}, "z": 1}]}))
        (self.root / "profile.json").write_text(json.dumps({
            "schema_version": 1, "height_m": .2, "grid_step": 2,
            "effectors": {"hand": {"part": "arm", "joints": 2, "start": [2, 6], "tip": [58, 6]}}}))
        (self.root / "set.json").write_text(json.dumps({
            "schema_version": 1, "world": {"strength": .3},
            "materials": {"wood": {"color": [.3, .2, .1], "roughness": .5}},
            "objects": [{"id": "desk", "kind": "box", "size": [1, 1, .04], "location": [0, -.02, 0], "material": "wood"},
                        {"id": "paper", "kind": "document", "width": .1, "height": .14, "location": [.1, 0, 0]}],
            "lights": [{"id": "key", "type": "AREA", "energy": 5, "location": [0, .6, .4], "rotation_deg": [-55, 0, 0]}]}))

    def spec(self, *, reach=(.05, .13, .03)):
        frames = self.FRAMES
        bake = {"schema_version": 1, "fps": {"num": 30, "den": 1}, "duration_frames": frames,
                "camera": {"position": repeat([0, .12, .8], frames), "orientation": repeat([0, 0, 0, 1], frames),
                           "fov": repeat(28, frames), "focus_distance": repeat(.8, frames)},
                "objects": {"paper": {"position": [[.1, 0, 0], [.11, .01, 0], [.12, .02, 0]],
                                      "orientation": repeat([0, 0, 0, 1], frames), "bend": [0, .1, .2]}},
                "lights": {"key": [1, .9, .8]}, "contacts": [{"frame": 1, "kind": "grasp", "actor": "a", "effector": "hand", "object": "paper"}],
                "actors": {"a": {"root": {"position": repeat([0, 0, 0], frames), "yaw": repeat(0, frames)},
                                 "parts": repeat({}, frames),
                                 "effectors": {"hand": {"position": repeat(list(reach), frames), "influence": [0, 1, 1], "holding": [False, True, True]}}}}}
        (self.root / "bake.json").write_text(json.dumps(bake))
        path = self.root / "shot.json"
        path.write_text(json.dumps({"schema_version": 1, "shot_id": "tiny", "set": "set.json", "bake": "bake.json",
                                    "actors": [{"id": "a", "pack": "pack", "profile": "profile.json"}],
                                    "render": {"profile": "draft", "width": 64, "height": 36}}))
        return path

    def run_build(self, spec: Path, output: Path) -> subprocess.CompletedProcess:
        return subprocess.run([BLENDER, "--background", "--factory-startup", "--python-exit-code", "1",
                               "--python", str(ROOT / "scripts/blender_entry.py"), "--", "build-shot",
                               "--spec", str(spec), "--output", str(output)],
                              capture_output=True, text=True, timeout=240)

    def test_renders_a_bundle_with_measured_contact(self):
        output = self.root / "out"
        result = self.run_build(self.spec(), output)
        self.assertEqual(result.returncode, 0, result.stdout[-1500:] + result.stderr[-1500:])
        manifest = json.loads((output / "shot-manifest.json").read_text())
        self.assertEqual(manifest["frames"], self.FRAMES)
        self.assertLess(manifest["contact_error_m"]["a.hand"], .002)
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames,width,height,codec_name", "-of", "json", str(output / "shot.mp4")],
                               capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual((stream["codec_name"], stream["width"], stream["height"], stream["nb_read_frames"]),
                         ("h264", 64, 36, str(self.FRAMES)))
        self.assertTrue((output / "shot.blend").exists())
        self.assertFalse(list(self.root.glob(".out.*")))

    def test_unreachable_target_fails_and_leaves_no_bundle(self):
        output = self.root / "out"
        result = self.run_build(self.spec(reach=(3, 3, 3)), output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Contact missed its target", result.stdout + result.stderr)
        self.assertFalse(output.exists())

    def test_describe_actor_reports_reach_in_y_up_metres(self):
        self.spec()
        result = subprocess.run([BLENDER, "--background", "--factory-startup", "--python-exit-code", "1",
                                 "--python", str(ROOT / "scripts/blender_entry.py"), "--", "describe-actor",
                                 "--pack", str(self.root / "pack"), "--profile", str(self.root / "profile.json")],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-800:] + result.stderr[-800:])
        line = next(item for item in result.stdout.splitlines() if item.startswith("{"))
        report = json.loads(line)
        hand = report["effectors"]["hand"]
        self.assertGreater(hand["reach_m"], .05)
        self.assertEqual(hand["part"], "arm")
        self.assertGreater(hand["rest"][1], .05, "Y must be height above the actor's feet")
        straight = sum((a - b) ** 2 for a, b in zip(hand["rest"], hand["shoulder"])) ** .5
        self.assertGreaterEqual(hand["reach_m"], straight * .98, "reach must include part scale")

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "out"
        output.mkdir()
        result = self.run_build(self.spec(), output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stdout + result.stderr)


if __name__ == "__main__":
    sys.exit(unittest.main())
