"""M1 source-pinned character region task queue and safe deterministic resume."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from kumiki_blender.character_build.__main__ import main as character_cli
from kumiki_blender.character_build.character_spec_v2 import plan_character
from kumiki_blender.character_build.region_batch import (
    make_region_batch, resume_region_batch, save_region_batch,
)


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


class RegionBatchTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root/"masks").mkdir()
        Image.new("L", (1600,1200), 0).save(self.root/"masks/front.png")
        self.spec = {
            "schema_version": 2, "character_id": "owl",
            "height_m": .35, "profile": "avian-biped-wings",
            "reference_frame": {"width":1600,"height":1200,"figure_px":1000},
            "regions": [
                {"id":"head","type":"head","source_confidence":"estimated"},
                {"id":"eye.L","type":"eye","parent":"head",
                 "source_confidence":"verified",
                 "reference_masks":{"front":"masks/front.png"},
                 "verified_views":["front"]},
                {"id":"eye.R","type":"eye","parent":"head","source_confidence":"estimated"},
                {"id":"lid.L","type":"eyelid","parent":"eye.L","source_confidence":"estimated"},
                {"id":"tail","type":"tail","source_confidence":"estimated"},
            ]
        }
        self.spec_path=self.root/"spec.json"
        self.spec_path.write_text(json.dumps(self.spec))
        plan_character(self.spec_path, self.root/"plan")
        self.batch_plan=save_region_batch(
            self.root/"plan", ["eye.L"], self.root/"batch", max_workers=1, memory_gib=24)
        self.baseline=self.root/"head.blend"
        self.baseline.write_bytes(b"BLENDER-head-approved")
        self.eye=self.root/"eye-L.blend"
        self.eye.write_bytes(b"BLENDER-eye-L-review")
        self.lid=self.root/"lid-L.blend"
        self.lid.write_bytes(b"BLENDER-lid-L-review")
        self.receipts={
            "schema_version":1,
            "batch_sha256":self.batch_plan["batch_sha256"],
            "preserved":{},
            "completed":[],
        }
        self.receipt_path=self.root/"receipts.json"

    def write_receipts(self):
        self.receipt_path.write_text(json.dumps(self.receipts))
        return self.receipt_path

    def complete_eye(self):
        self.receipts["completed"].append({
            "region":"eye.L","artifact":self.eye.name,"sha256":sha(self.eye),
            "dependency_sha256":{"head":sha(self.baseline)}
        })

    def complete_lid(self):
        self.receipts["completed"].append({
            "region":"lid.L","artifact":self.lid.name,"sha256":sha(self.lid),
            "dependency_sha256":{"eye.L":sha(self.eye)}
        })

    def preserve_head(self):
        self.receipts["preserved"]["head"]={
            "artifact":self.baseline.name,"sha256":sha(self.baseline)
        }

    def test_only_changed_region_and_dependents_are_scheduled(self):
        p=self.batch_plan
        self.assertEqual(p["change_roots"],["eye.L"])
        self.assertEqual(p["scheduled"],["eye.L","lid.L"])
        self.assertEqual(p["preserve_unchanged"],["head","eye.R","tail"])
        self.assertEqual(p["theoretical_waves"],[["eye.L"],["lid.L"]])
        tasks={t["region"]:t for t in p["tasks"]}
        self.assertEqual(tasks["eye.L"]["preserved_dependencies"],["head"])
        self.assertEqual(tasks["lid.L"]["scheduled_dependencies"],["eye.L"])
        self.assertEqual(tasks["eye.L"]["verified_source_views"],["front"])
        self.assertTrue((self.root/"batch/region-batch-plan.json").is_file())
        with self.assertRaises(FileExistsError):
            save_region_batch(self.root/"plan", ["eye.L"], self.root/"batch")

    def test_missing_preserved_parent_blocks_work_until_approved(self):
        result=resume_region_batch(
            self.root/"batch/region-batch-plan.json",None,self.root/"blocked")
        self.assertEqual(result["status"],"blocked-by-upstream-evidence")
        self.assertEqual(result["ready_for_blender"],[])
        self.assertEqual(result["waiting_for_dependencies"]["eye.L"],["head"])
        self.assertEqual(result["heavy_blender_executions_started"],0)
        self.assertTrue(result["artistic_and_deformation_review_required"])

    def test_resume_schedules_only_unblocked_tasks_and_never_runs_blender(self):
        self.preserve_head()
        self.write_receipts()
        plan=self.root/"batch/region-batch-plan.json"
        first=resume_region_batch(plan,self.receipt_path,self.root/"step1")
        self.assertEqual(first["ready_for_blender"],["eye.L"])
        self.assertIn("lid.L",first["waiting_for_dependencies"])
        self.complete_eye()
        self.write_receipts()
        second=resume_region_batch(plan,self.receipt_path,self.root/"step2")
        self.assertEqual(second["completed"],["eye.L"])
        self.assertEqual(second["ready_for_blender"],["lid.L"])
        self.complete_lid()
        self.write_receipts()
        final=resume_region_batch(plan,self.receipt_path,self.root/"step3")
        self.assertEqual(final["status"],"complete-awaiting-source-and-animation-qa")
        self.assertEqual(final["ready_for_blender"],[])
        self.assertEqual(final["completed"],["eye.L","lid.L"])
        self.assertTrue((self.root/"step3/batch-status.json").is_file())

    def test_rejects_stale_source_art_and_plan_mutation(self):
        with Image.open(self.root/"masks/front.png") as picture:
            changed=picture.copy()
        changed.putpixel((0,0),255)
        changed.save(self.root/"masks/front.png")
        with self.assertRaisesRegex(ValueError,"changed"):
            resume_region_batch(self.root/"batch/region-batch-plan.json",
                                None,self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_dependency_sha_breaks_resume_when_eye_or_parent_changes(self):
        self.preserve_head()
        self.complete_eye()
        self.write_receipts()
        self.eye.write_bytes(b"BLENDER-a-different-eye")
        with self.assertRaisesRegex(ValueError,"artifact changed"):
            resume_region_batch(self.root/"batch/region-batch-plan.json",
                                self.receipt_path,self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_broken_upstream_dependency_blocks_completed_child_receipt(self):
        self.preserve_head()
        self.complete_lid()
        self.write_receipts()
        with self.assertRaisesRegex(ValueError,"upstream dependency"):
            resume_region_batch(self.root/"batch/region-batch-plan.json",
                                self.receipt_path,self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_original_head_change_invalidates_every_dependent(self):
        result=make_region_batch(self.root/"plan",["head"])
        self.assertEqual(result["scheduled"],["head","eye.L","eye.R","lid.L"])
        self.assertEqual(result["preserve_unchanged"],["tail"])
        self.assertEqual(result["theoretical_waves"],
                         [["head"],["eye.L"],["eye.R"],["lid.L"]])

    def test_memory_budget_rejects_parallel_blender_on_24gb(self):
        with self.assertRaisesRegex(ValueError,"only supports 1"):
            make_region_batch(self.root/"plan",["eye.L","eye.R"],
                              max_workers=2,memory_gib=24)
        plan=make_region_batch(self.root/"plan",["eye.L","eye.R"],
                               max_workers=2,memory_gib=32)
        self.assertEqual(plan["theoretical_waves"],[["eye.L","eye.R"],["lid.L"]])
        self.assertEqual(plan["max_heavy_blender_workers"],2)
        with self.assertRaisesRegex(ValueError,"max-workers"):
            make_region_batch(self.root/"plan",["eye.L"],max_workers=0)

    def test_conflicting_resume_batch_and_duplicates_are_rejected(self):
        self.preserve_head()
        self.complete_eye()
        self.receipts["batch_sha256"]="0"*64
        self.write_receipts()
        with self.assertRaisesRegex(ValueError,"do not belong"):
            resume_region_batch(self.root/"batch/region-batch-plan.json",
                                self.receipt_path,self.root/"invalid")
        self.receipts["batch_sha256"]=self.batch_plan["batch_sha256"]
        self.complete_eye()
        self.write_receipts()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            resume_region_batch(self.root/"batch/region-batch-plan.json",
                                self.receipt_path,self.root/"invalid")

    def test_cli_plan_and_status_are_machine_readable(self):
        out=io.StringIO()
        with redirect_stdout(out):
            code=character_cli([
                "plan-region-batch","--character",str(self.root/"plan"),
                "--changed","eye.L","--output",str(self.root/"cli-plan")
            ])
        self.assertEqual(code,0)
        self.assertEqual(json.loads(out.getvalue())["scheduled"],["eye.L","lid.L"])
        report=io.StringIO()
        with redirect_stdout(report):
            code=character_cli([
                "resume-region-batch","--plan",
                str(self.root/"cli-plan/region-batch-plan.json"),
                "--output",str(self.root/"cli-status"),
            ])
        self.assertEqual(code,1)
        self.assertEqual(json.loads(report.getvalue())["status"],
                         "blocked-by-upstream-evidence")
        self.assertFalse((self.root/"cli-status"/"candidate.blend").exists())


if __name__=="__main__":
    unittest.main()
