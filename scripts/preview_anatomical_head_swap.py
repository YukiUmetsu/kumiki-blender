"""Generate an anatomical full-figure GLB preview with geometry preservation QA.

Preferred one-command workflow:
  python scripts/preview_anatomical_head_swap.py \
    --base .review/christ_redeemer/r50.glb \
    --blender /path/to/blender \
    --output .review/christ_redeemer/anatomical-HOLD.glb \
    --views .review/christ_redeemer/anatomical-views

Alternatively, --donor <head-study.glb> uses an existing Blender study.
Use --alignment bounds ONLY for a generic donor export in different units;
this is not photo or scan registration. No preview is production-approved.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kumiki_blender.head_graft import graft_anatomical_head


def blender_study_command(blender, destination, *, carved_hair=False):
    """Use the existing CC0 anatomy head-study generator as the donor."""
    command = [str(blender), "-b", "--factory-startup", "--python-exit-code", "1",
               "--python", str(ROOT / "scripts/landmarks/christ_redeemer.py"), "--",
               "--head-study", "--face-method", "anatomical", "--verify",
               "--output", str(destination)]
    if carved_hair:
        command.append("--carved-hair")
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base",type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--donor",type=Path,help="Existing anatomical head-study GLB")
    source.add_argument("--blender",type=Path,help="Generate the CC0 anatomical donor with Blender CLI")
    p.add_argument("--alignment",choices=("world","bounds"),default="world")
    p.add_argument("--replace-hair",action="store_true",
                   help="Only if donor includes a separately reviewed hair study")
    p.add_argument("--carved-hair",action="store_true",
                   help="Experimental carved hair; requires --blender --replace-hair")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--metrics",type=Path)
    p.add_argument("--views",type=Path,help="Optionally render four head views using VTK")
    args=p.parse_args()
    if args.base.resolve()==args.output.resolve():
        p.error("Preview output must not overwrite the approved full-figure GLB")
    if args.blender and args.alignment!="world":
        p.error("Blender-generated head studies already have world coordinates")
    if args.carved_hair and (not args.blender or not args.replace_hair):
        p.error("--carved-hair requires --blender and --replace-hair")
    import trimesh
    with tempfile.TemporaryDirectory(prefix="christ-head-graft-") as tmp:
        donor_file=args.donor
        if args.blender:
            donor_file=Path(tmp)/"anatomical_head_study.glb"
            subprocess.run(blender_study_command(args.blender,donor_file,
                           carved_hair=args.carved_hair),cwd=ROOT,check=True)
        base=trimesh.load(args.base,force="scene")
        donor=trimesh.load(donor_file,force="scene")
        candidate,audit=graft_anatomical_head(base,donor,alignment=args.alignment,
                                              replace_hair=args.replace_hair)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        candidate.export(args.output,file_type="glb")
        metrics=args.metrics or args.output.with_suffix(".json")
        metrics.parent.mkdir(parents=True,exist_ok=True)
        metrics.write_text(json.dumps(audit,indent=2)+"\n")
        if args.views:
            subprocess.run([sys.executable,str(ROOT/"scripts/render_glb_head_qa.py"),
                            str(args.output),"--output",str(args.views)],
                           cwd=ROOT,check=True)
        print("ANATOMICAL_HEAD_PREVIEW "+json.dumps({"status":audit["status"],
            "head_vertices":audit["donor_head_vertices"],"output":str(args.output),
            "metrics":str(metrics),"views":str(args.views) if args.views else None}))


if __name__=="__main__":main()
