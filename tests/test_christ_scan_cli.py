"""End-to-end rights-pinned synthetic GLB alignment; NOT a monument scan."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]
CLI=ROOT/'scripts/landmarks/align_christ_scan.py'


class ChristScanCLIIntegration(unittest.TestCase):
    def test_rights_pinned_synthetic_glb_alignment_and_replica_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory)
            mesh=trimesh.creation.icosphere(subdivisions=3,radius=1)
            source=tmp/'synthetic.glb'
            mesh.export(source)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            record={"schema_version":1,"asset_id":"christ-synthetic-fixture",
                    "creator":"Unit test","source_url":"https://example.org/christ-synthetic-fixture",
                    "license":"CC0-1.0","attribution":"","redistribution_allowed":True,
                    "commercial_use_allowed":True,"expected_sha256":digest}
            provenance=tmp/'provenance.json'
            provenance.write_text(json.dumps(record))
            rng=np.random.default_rng(7)
            ids=rng.choice(len(mesh.vertices),size=16,replace=False)
            src=mesh.vertices[ids]
            target=src*1.2+[0,35,1.]
            anchor_file=tmp/'anchors.json'
            annotations={"schema_version":1,"coordinate_system":"glb-y-up-metres",
                         "source_kind":"actual-monument-scan",
                         "source_authenticity_review":"SYNTHETIC TEST, NOT A REAL MONUMENT",
                         "source_sha256":digest,
                         "landmarks":[{"name":f"test-{i}","source_xyz":p.tolist(),
                                       "target_xyz":q.tolist()} for i,(p,q) in enumerate(zip(src,target))]}
            anchor_file.write_text(json.dumps(annotations))
            out=tmp/'aligned-HOLD.glb';audit=tmp/'audit.json'
            cmd=[sys.executable,str(CLI),'--source',str(source),'--provenance',str(provenance),
                 '--anchors',str(anchor_file),'--output',str(out),'--audit',str(audit)]
            env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
            result=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(out.is_file())
            report=json.loads(audit.read_text())
            self.assertFalse(report['production_approved'])
            self.assertLess(report['rms_inlier_m'],.001)
            aligned=trimesh.load(out,force='scene')
            self.assertAlmostEqual(float(aligned.bounds[0][1]),
                                   float(mesh.bounds[0][1])*1.2+35,delta=.01)
            annotations['source_kind']='replica'
            anchor_file.write_text(json.dumps(annotations))
            out.unlink()
            result=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
