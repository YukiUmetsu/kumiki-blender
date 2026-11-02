"""Splat camera/video integration: real FFmpeg, mocked GPU rasterizer only."""

from __future__ import annotations

import contextlib
import io
import json
import math
import hashlib
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.cli import main
from kumiki_blender.reconstruction.references import ReferenceError
from kumiki_blender.reconstruction.splat_video import (
    SplatVideoSettings, _position_bounds, make_orbit_track, plan_splat_video,
    render_splat_video)


class SplatVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ply = self.root / "splat.ply"
        fields = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 400\n"
        for name in fields:
            header += f"property float {name}\n"
        header += "end_header\n"
        rows = []
        for index in range(400):
            # A small, camera-visible statue-sized volume and two outliers.
            x = ((index % 20) - 10) / 10
            y = (index // 20) / 10
            z = ((index * 7) % 19 - 9) / 10
            if index in (0, 399):
                x = 1000.0
            rows.append(struct.pack("<14f", x, y, z, *([0.5] * 11)))
        self.ply.write_bytes(header.encode("ascii") + b"".join(rows))
        self.rasterizer = self.root / "fake-splat-transform"
        self.rasterizer.write_text(f'''#!{sys.executable}
import json, pathlib, sys
from PIL import Image
input_file, output = sys.argv[1:3]
assert pathlib.Path(input_file).is_file()
output = pathlib.Path(output)
if output.suffix == '.html':
    output.write_text('<html>private viewer</html>')
    sys.exit(0)
if 'fail-render' in input_file:
    sys.exit(9)
track = json.loads(pathlib.Path(sys.argv[sys.argv.index('--camera-track')+1]).read_text())
width, height = map(int, sys.argv[sys.argv.index('--resolution')+1].split('x'))
for index, frame in enumerate(track['frames']):
    if 'incomplete' in input_file and index == 4:
        continue
    Image.new('RGB', (width, height), (index % 256, 40, 80)).save(
        output.with_name('frame.%04d.webp' % index), lossless=True)
''')
        self.rasterizer.chmod(0o755)

    def settings(self, **overrides):
        return SplatVideoSettings(fps=24, seconds=2, width=256, height=256,
                                  splat_transform=str(self.rasterizer), **overrides)

    def test_camera_sample_trims_extreme_outliers(self):
        center, extent = _position_bounds(self.ply)
        self.assertLess(abs(center[0]), 1)
        self.assertLess(extent, 4)
        self.assertGreater(extent, 1)

    def test_orbit_eases_and_keeps_camera_looking_at_subject(self):
        track = make_orbit_track((0, 1, 0), 3, self.settings())
        self.assertEqual(track["frameRate"], 24)
        self.assertEqual(len(track["frames"]), 48)
        self.assertEqual(track["frames"][0]["position"], track["frames"][1]["position"])
        self.assertEqual(track["frames"][-1]["position"], track["frames"][-2]["position"])
        self.assertNotEqual(track["frames"][10]["position"], track["frames"][30]["position"])
        for frame in track["frames"]:
            self.assertEqual(frame["target"], [0, 1, 0])
            self.assertAlmostEqual(math.dist(frame["position"], [0, 1, 0]), 3, places=5)

    def test_preflight_does_not_create_files_and_rejects_existing_revision(self):
        output = self.root / "video-r01"
        report = plan_splat_video(self.ply, output, self.settings())
        self.assertIn("preflight", report["status"])
        self.assertFalse(output.exists())
        output.mkdir()
        with self.assertRaisesRegex(ReferenceError, "exists"):
            plan_splat_video(self.ply, output, self.settings())

    def test_rejects_symlinked_input_and_failed_revision(self):
        linked = self.root / "linked.ply"
        linked.symlink_to(self.ply)
        with self.assertRaisesRegex(ReferenceError, "symlink"):
            plan_splat_video(linked, self.root / "out", self.settings())
        source = self.root / "failed-revision"
        source.mkdir()
        (source / "splat.ply").write_bytes(self.ply.read_bytes())
        (source / "splat-report.json").write_text(json.dumps({"status": "failed"}))
        with self.assertRaisesRegex(ReferenceError, "not passed"):
            plan_splat_video(source, self.root / "out", self.settings())
        (source / "splat-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required",
            "artifact": {"sha256": "0" * 64}}))
        with self.assertRaisesRegex(ReferenceError, "changed"):
            plan_splat_video(source, self.root / "out", self.settings())
        (source / "splat-report.json").write_text(json.dumps({
            "status": "candidate / visual-review-required",
            "artifact": {"sha256": hashlib.sha256(self.ply.read_bytes()).hexdigest()}}))
        plan_splat_video(source, self.root / "out", self.settings())

    def test_track_mismatch_fails_before_creating_revision(self):
        track = self.root / "track.json"
        track.write_text(json.dumps({"frameRate": 30, "frames": [{"position": [0, 1, 2]}]}))
        with self.assertRaisesRegex(ReferenceError, "must match"):
            plan_splat_video(self.ply, self.root / "out", self.settings(), camera_track=track)
        self.assertFalse((self.root / "out").exists())
        track.write_text(json.dumps({"frameRate": 24, "frames": "broken-not-an-array"}))
        with self.assertRaisesRegex(ReferenceError, "must match"):
            plan_splat_video(self.ply, self.root / "out", self.settings(), camera_track=track)
        track.write_text(json.dumps({"frameRate": 24, "frames": [{"position": [0, 0, 0]}] * 48}))
        with self.assertRaisesRegex(ReferenceError, "target"):
            plan_splat_video(self.ply, self.root / "out", self.settings(), camera_track=track)

    def test_spz_orbit_requires_manual_framing(self):
        spz = self.root / "test.spz"
        spz.write_bytes(struct.pack("<IIIBBBBI12s", 0x5053474e, 4, 2, 3, 12, 0, 1, 32,
                                     bytes(12)) + struct.pack("<QQ", 4, 24) + b"data")
        with self.assertRaisesRegex(ReferenceError, "SPZ orbit"):
            plan_splat_video(spz, self.root / "out", self.settings())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_real_mp4_encode_and_viewer_handoff(self):
        output = self.root / "r01"
        report = render_splat_video(self.ply, output, self.settings(viewer=True))
        self.assertTrue(report["status"].startswith("candidate"), report)
        self.assertEqual(report["video"]["encoded_frames"], 48)
        self.assertEqual(report["video"]["codec"], "h264")
        self.assertEqual(report["video"]["width"], 256)
        self.assertTrue((output / "video.mp4").is_file())
        self.assertTrue((output / "viewer.html").is_file())
        self.assertTrue((output / "logs/decode-audit.log").is_file())
        self.assertTrue((output / "contact-sheet.jpg").is_file())
        self.assertEqual(len(list((output / "frames").glob("*.webp"))), 48)
        self.assertEqual(len(json.loads((output / "camera-track.json").read_text())["frames"]), 48)
        self.assertFalse(report["distribution_allowed"])
        with self.assertRaisesRegex(ReferenceError, "exists"):
            render_splat_video(self.ply, output, self.settings())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_external_camera_track_is_snapshotted_into_immutable_revision(self):
        track = self.root / "authored.json"
        data = make_orbit_track((0, 0, 0), 4, self.settings())
        track.write_text(json.dumps(data))
        output = self.root / "authored-video"
        report = render_splat_video(self.ply, output, self.settings(), camera_track=track)
        self.assertTrue(report["status"].startswith("candidate"))
        self.assertEqual(json.loads((output / "camera-track.json").read_text()), data)
        track.write_text("modified-after-render")
        self.assertEqual(json.loads((output / "camera-track.json").read_text()), data)

    def test_frame_sequence_failures_are_recorded_without_claiming_success(self):
        missing = self.root / "incomplete.ply"
        missing.write_bytes(self.ply.read_bytes())
        output = self.root / "missing-frames"
        report = render_splat_video(missing, output, self.settings())
        self.assertEqual(report["failed_stage"], "frame-sequence")
        self.assertEqual(report["actual_frames"], 47)
        self.assertFalse((output / "video.mp4").exists())
        self.assertTrue((output / "video-report.json").exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_cli_plan_then_render(self):
        output = self.root / "cli-r01"
        common = ["--splat", str(self.ply), "--output", str(output),
                  "--splat-transform", str(self.rasterizer),
                  "--fps", "24", "--seconds", "2", "--width", "256", "--height", "256"]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = main(["plan-splat-video", *common])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(printed.getvalue())["fps"], 24)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = main(["render-splat-video", *common])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(printed.getvalue())["video"]["encoded_frames"], 48)


if __name__ == "__main__":
    unittest.main()
