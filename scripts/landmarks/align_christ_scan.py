"""Align a locally supplied, rights-cleared Christ monument scan by 3D anchors.

  python scripts/landmarks/align_christ_scan.py \
    --source source.glb --provenance source-license.json \
    --anchors christ-source-to-r50-landmarks.json \
    --output .review/christ/registered-source-HOLD.glb \
    --audit .review/christ/registered-source-HOLD.json

This never downloads models, promotes production geometry, or treats a
generic anatomy donor/replica as an authenticated monument scan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

from kumiki_blender.asset_import import load_asset_provenance,inspect_source_glb
from kumiki_blender.scan_registration import fit_scan_landmarks


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True,
                        help='Verified commercial + redistribution rights (asset_import schema v1)')
    parser.add_argument('--anchors',type=Path,required=True,
                        help='JSON with distinct named source_xyz and target_xyz 3D anchors')
    parser.add_argument('--output',type=Path,required=True,
                        help='Experimental standalone GLB, never production full figure')
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args()
    if len({args.source.resolve(),args.output.resolve(),args.audit.resolve()})!=3:
        parser.error('Output, audit and input must be different files')
    provenance=load_asset_provenance(args.provenance)
    if provenance.commercial_use_allowed is not True:
        parser.error('Commercial use must be explicitly verified before exporting a scan')
    inspection=inspect_source_glb(args.source,provenance)
    data=json.loads(args.anchors.read_text(encoding='utf-8'))
    if data.get('schema_version')!=1 or data.get('coordinate_system')!='glb-y-up-metres':
        parser.error('Anchors must declare schema_version 1 and glb-y-up-metres coordinates')
    if data.get('source_sha256')!=inspection['input_sha256']:
        parser.error('Anchor file must pin the exact scan GLB SHA-256')
    if data.get('source_kind')!='actual-monument-scan':
        parser.error('Reference donor/replica cannot be exported as an authenticated monument scan')
    fit=fit_scan_landmarks(data['landmarks'])
    import numpy as np
    import trimesh
    scene=trimesh.load(args.source,force='scene')
    scene.apply_transform(np.asarray(fit['matrix_source_to_target']))
    audit={**fit,'source_kind':data['source_kind'],
           'source_authenticity_review':data.get('source_authenticity_review','unverified'),
           'provenance':provenance.to_manifest(inspection['input_sha256']),
           'source_meshes':len(scene.geometry),
           'aligned_bounds_m':np.round(scene.bounds,5).tolist(),
           'production_approved':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    scene.export(args.output,file_type='glb')
    args.audit.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print('CHRIST_SCAN_REGISTERED_HOLD',json.dumps({'rms_m':fit['rms_inlier_m'],
         'inliers':fit['inlier_count'],'total':fit['total_count'],
         'output':str(args.output),'audit':str(args.audit)}))


if __name__=='__main__':main()
