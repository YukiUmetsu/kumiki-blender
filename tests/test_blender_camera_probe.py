"""Live Blender QA: projection and raycast must agree in four head cameras."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
GENERATOR=ROOT/"scripts/landmarks/christ_redeemer.py"
PROBE=ROOT/"scripts/landmarks/probe_face_anchor_pixels.py"


@unittest.skipUnless(os.environ.get("KUMIKI_BLENDER_BINARY"),
                     "Set KUMIKI_BLENDER_BINARY to test Blender evaluated-mesh raycasts")
class BlenderCameraProbeTests(unittest.TestCase):
    def test_four_cameras_hit_real_anatomical_mesh_and_reproject(self):
        blender=os.environ["KUMIKI_BLENDER_BINARY"]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            blend=root/"head.blend"
            build=subprocess.run([blender,"-b","--factory-startup",
                "--python-exit-code","1","--python",str(GENERATOR),"--",
                "--head-study","--face-method","anatomical","--blend",str(blend)],
                text=True,capture_output=True,timeout=100)
            self.assertEqual(build.returncode,0,build.stdout[-1000:]+build.stderr[-1000:])
            for shot in ("head_front","head_profile","head_three_quarter","head_low"):
                with self.subTest(shot=shot):
                    source=root/(shot+".json")
                    source.write_text(json.dumps({"model_shot":shot,
                        "model_image_size":[700,700],"image_size":[700,700],
                        "subject_photo_url":"https://example.org/reference.jpg",
                        "model_revision":"qa-test",
                        "landmarks":[{"name":"head_center",
                                      "model_uv":[350,350],"photo_uv":[350,350]}]}))
                    target=root/(shot+"-3d.json")
                    result=subprocess.run([blender,"-b",str(blend),
                        "--python-exit-code","1","--python",str(PROBE),"--",
                        "--input",str(source),"--output",str(target)],
                        text=True,capture_output=True,timeout=50)
                    self.assertEqual(result.returncode,0,
                                     result.stdout[-1200:]+result.stderr[-1200:])
                    data=json.loads(target.read_text())
                    self.assertEqual(data["model_shot"],shot)
                    self.assertEqual(len(data["landmarks"]),1)
                    self.assertLess(data["landmarks"][0]["model_reprojection_error_px"],.6)
                    self.assertFalse(data["landmarks"][0]["occluded"])


if __name__=="__main__":
    unittest.main()
