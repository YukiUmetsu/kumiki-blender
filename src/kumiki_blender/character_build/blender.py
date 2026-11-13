"""Run the character_build Blender workers headless.

Each worker is a script in this package that Blender executes with
``--python``. Arguments go in one JSON file, so paths with spaces and lists
need no shell quoting. A worker writes ``report.json`` into its output folder.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKERS = {"setup": "blender_setup.py", "assemble": "blender_assemble.py", "retopo": "blender_retopo.py",
           "rig": "blender_rig.py", "pose-qa": "blender_pose_qa.py", "penetration-repair": "blender_penetration_repair.py", "region-sculpt": "blender_region_sculpt.py", "geodesic-fit": "blender_landmark_fit.py", "rig-controls": "blender_rig_controls.py", "topology-qa": "blender_topology_qa.py", "joint-refine": "blender_joint_refine.py", "animation-bake": "blender_animation_bake.py", "visemes": "blender_visemes.py", "anatomy": "blender_anatomy.py", "performance": "blender_performance.py", "swap": "blender_swap.py", "fur": "blender_fur.py"}


def find_blender(explicit: str | None = None) -> str:
    for candidate in (explicit, shutil.which("blender"), "/Applications/Blender.app/Contents/MacOS/Blender"):
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Blender not found. Pass --blender /path/to/blender")


def run_worker(worker: str, arguments: dict, *, blender: str | None = None, blend_file: Path | None = None,
               timeout_s: int = 3600) -> dict:
    """Run one worker. Return its report. Raise with Blender's output when it fails."""
    out = Path(arguments["output"])
    with tempfile.TemporaryDirectory() as tmp:
        args_file = Path(tmp) / "arguments.json"
        args_file.write_text(json.dumps(arguments), encoding="utf-8")
        command = [find_blender(blender), "-b", "--python-exit-code", "1"]
        command += [str(blend_file)] if blend_file else ["--factory-startup"]
        command += ["--python", str(HERE / WORKERS[worker]), "--", str(args_file)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    report = out / "report.json"
    if result.returncode != 0 or not report.is_file():
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
        raise RuntimeError(f"Blender worker {worker} failed (exit {result.returncode}):\n{tail}")
    return json.loads(report.read_text(encoding="utf-8"))
