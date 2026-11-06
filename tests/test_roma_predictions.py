"""Synthetic RoMa runner tests; no model download or GPU is required."""
import hashlib
import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kumiki_blender.reconstruction.roma_predictions import generate_roma_predictions
from kumiki_blender.reconstruction.roma_candidate_database import prepare_roma_candidate_database


class StubPredictor:
    def __init__(self, *, disagree=False):
        self.disagree = disagree
        self.pairs = []

    def predict(self, anchor, target, pixels, anchor_size):
        self.pairs.append((anchor.name, target.name))
        delta = 0.5 if self.disagree and anchor.name == 'b.png' else 0
        return [([2 * (30 + delta) / 100 - 1, 2 * 40 / 100 - 1], 0.75) for _ in pixels]


class PredictionGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        (self.source / 'sparse/0').mkdir(parents=True)
        (self.source / 'text-models/0').mkdir(parents=True)
        (self.source / 'images').mkdir()
        names = ['a.png', 'b.png', 'c.png', 'target.png', 'rear.png']
        entries = []
        for name in names:
            file = self.source / 'images' / name
            Image.new('RGB', (100, 100), (90, 70, 50)).save(file)
            entries.append({'filename': name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
        (self.source / 'staged-reference-index.json').write_text(json.dumps({'images': entries}))
        (self.source / 'text-models/0/images.txt').write_text(''.join(
            f'{i} 1 0 0 0 0 0 0 1 {name}\n10 10 10\n' for i, name in enumerate(names[:3], 1)))
        (self.source / 'text-models/0/points3D.txt').write_text(
            '10 1 2 3 255 255 255 0.1 1 0 2 0 3 0\n')
        with sqlite3.connect(self.source / 'database.db') as db:
            db.executescript('CREATE TABLE images(image_id INTEGER,name TEXT,camera_id INTEGER); '
                             'CREATE TABLE cameras(camera_id INTEGER,width INTEGER,height INTEGER); '
                             'CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB); '
                             'CREATE TABLE descriptors(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB); '
                             'CREATE TABLE matches(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB); '
                             'CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB,config INTEGER);')
            db.executemany('INSERT INTO images VALUES(?,?,1)', [(i, name) for i, name in enumerate(names, 1)])
            db.execute('INSERT INTO cameras VALUES(1,100,100)')
            for i in (4, 5):
                db.execute('INSERT INTO keypoints VALUES(?,1,2,?)', (i, struct.pack('<ff', 2, 2)))
                db.execute('INSERT INTO descriptors VALUES(?,1,2,?)', (i, b'12'))

    def test_generates_multiple_targets_without_mutating_source(self):
        before = hashlib.sha256((self.source / 'database.db').read_bytes()).hexdigest()
        stub = StubPredictor()
        report = generate_roma_predictions(self.source, self.root / 'predictions',
                                           targets=['target.png', 'rear.png'],
                                           anchors=['a.png', 'b.png', 'c.png'], predictor=stub)
        self.assertEqual(len(stub.pairs), 6)
        self.assertEqual(report['targets']['target.png']['agreed_tracks'], 1)
        evidence = self.root / 'predictions/target.png.roma.json'
        result = prepare_roma_candidate_database(self.source, evidence, self.root / 'candidate')
        self.assertEqual(result['candidate_tracks'], 1)
        self.assertEqual(hashlib.sha256((self.source / 'database.db').read_bytes()).hexdigest(), before)
        self.assertEqual(len(list((self.root / 'predictions').glob('*.roma.json'))), 2)

    def test_import_rejects_predictions_pinned_to_different_target_photo(self):
        generate_roma_predictions(self.source, self.root / 'predictions',
                                  targets=['target.png'], anchors=['a.png', 'b.png'],
                                  predictor=StubPredictor())
        evidence = self.root / 'predictions/target.png.roma.json'
        payload = json.loads(evidence.read_text())
        payload['source_photo_sha256'] = '0' * 64
        evidence.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'different target photo revision'):
            prepare_roma_candidate_database(self.source, evidence, self.root / 'invalid')
        self.assertFalse((self.root / 'invalid').exists())

    def test_refuses_registered_target_and_output_inside_source(self):
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            generate_roma_predictions(self.source, self.root / 'invalid',
                                      targets=['a.png'], anchors=['a.png', 'b.png'],
                                      predictor=StubPredictor())
        with self.assertRaisesRegex(ValueError, 'separate'):
            generate_roma_predictions(self.source, self.source / 'nested',
                                      targets=['target.png'], anchors=['a.png', 'b.png'],
                                      predictor=StubPredictor())
        self.assertFalse((self.source / 'nested').exists())

    def test_fails_closed_on_changed_photo_before_model_inference(self):
        Image.new('RGB', (100, 100), 'red').save(self.source / 'images/target.png')
        predictor = StubPredictor()
        with self.assertRaisesRegex(ValueError, 'modified staged image'):
            generate_roma_predictions(self.source, self.root / 'invalid',
                                      targets=['target.png'], anchors=['a.png', 'b.png'],
                                      predictor=predictor)
        self.assertEqual(predictor.pairs, [])

    def test_subject_background_disallows_predictions(self):
        qa = self.root / 'target-qa.png'
        Image.new('L', (100, 100), 0).save(qa)
        index = {'schema': 'subject-feature-index-v1', 'images': {'target.png': {
            'confidence_path': str(qa), 'confidence_sha256': hashlib.sha256(qa.read_bytes()).hexdigest()}}}
        (self.source / 'subject-feature-index.json').write_text(json.dumps(index))
        report = generate_roma_predictions(self.source, self.root / 'predictions',
                                           targets=['target.png'], anchors=['a.png', 'b.png'],
                                           predictor=StubPredictor())
        self.assertEqual(report['targets']['target.png']['agreed_tracks'], 0)
        self.assertEqual(report['targets']['target.png']['predictions'], 0)

    def test_staged_photo_changed_during_long_inference_cannot_be_receipted(self):
        class ChangeDuringInference(StubPredictor):
            def predict(self, anchor, target, pixels, anchor_size):
                Image.new('RGB', (100, 100), 'purple').save(target)
                return super().predict(anchor, target, pixels, anchor_size)
        output = self.root / 'unstable'
        with self.assertRaisesRegex(ValueError, 'Changed staged photo during RoMa'):
            generate_roma_predictions(self.source, output,
                                      targets=['target.png'], anchors=['a.png', 'b.png'],
                                      predictor=ChangeDuringInference())
        self.assertFalse((output / 'target.png.roma.json').exists())
        self.assertEqual(json.loads((output / 'roma-report.json').read_text())['targets'], {})

    def test_partial_batch_keeps_completed_target_receipts(self):
        class InterruptedPredictor(StubPredictor):
            def predict(self, anchor, target, pixels, anchor_size):
                if target.name == 'rear.png':
                    raise RuntimeError('interrupted experiment')
                return super().predict(anchor, target, pixels, anchor_size)
        output = self.root / 'partial'
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            generate_roma_predictions(self.source, output,
                                      targets=['target.png', 'rear.png'],
                                      anchors=['a.png', 'b.png'], predictor=InterruptedPredictor())
        self.assertTrue((output / 'target.png.roma.json').is_file())
        report = json.loads((output / 'roma-report.json').read_text())
        self.assertEqual(report['status'], 'running-unverified-roma-predictions')
        self.assertIn('target.png', report['targets'])
        self.assertNotIn('rear.png', report['targets'])

        # No re-inference for the completed target: its hashed predictions
        # survive the interruption byte-for-byte.
        first = output / 'target.png.roma.json'
        before = hashlib.sha256(first.read_bytes()).hexdigest()
        resumed = StubPredictor()
        result = generate_roma_predictions(self.source, output, resume=True,
                                           targets=['target.png', 'rear.png'],
                                           anchors=['a.png', 'b.png'], predictor=resumed)
        self.assertEqual(resumed.pairs, [('a.png', 'rear.png'), ('b.png', 'rear.png')])
        self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), before)
        self.assertEqual(result['status'], 'unverified-roma-predictions')
        self.assertEqual(result['resumed_completed_targets'], ['target.png'])
        self.assertEqual(set(result['targets']), {'target.png', 'rear.png'})
        # A complete resume must not load RoMa, even if optional deps are absent.
        again = generate_roma_predictions(self.source, output, resume=True,
                                          targets=['target.png', 'rear.png'],
                                          anchors=['a.png', 'b.png'])
        self.assertEqual(again['status'], 'unverified-roma-predictions')

    def test_resume_refuses_changed_source_configuration_and_completed_payload(self):
        output = self.root / 'predictions'
        params = dict(targets=['target.png'], anchors=['a.png', 'b.png'])
        generate_roma_predictions(self.source, output, predictor=StubPredictor(), **params)
        existing = output / 'target.png.roma.json'
        before = existing.read_bytes()
        with self.assertRaisesRegex(ValueError, 'mismatch for min_confidence'):
            generate_roma_predictions(self.source, output, resume=True,
                                      min_confidence=0.2, predictor=StubPredictor(), **params)
        with self.assertRaisesRegex(ValueError, 'mismatch for requested_targets'):
            generate_roma_predictions(self.source, output, resume=True,
                                      targets=['target.png', 'rear.png'],
                                      anchors=['a.png', 'b.png'], predictor=StubPredictor())
        self.assertEqual(existing.read_bytes(), before)

        with sqlite3.connect(self.source / 'database.db') as db:
            db.execute('UPDATE cameras SET width=101 WHERE camera_id=1')
        with self.assertRaisesRegex(ValueError, 'mismatch for source_sha256'):
            generate_roma_predictions(self.source, output, resume=True,
                                      predictor=StubPredictor(), **params)
        with sqlite3.connect(self.source / 'database.db') as db:
            db.execute('UPDATE cameras SET width=100 WHERE camera_id=1')
        # SQLite may not return to its byte-identical file hash after UPDATE.
        # Use a fresh isolated source/output to check payload tampering below.

    def test_resume_rejects_modified_prediction_without_overwriting_it(self):
        output = self.root / 'predictions'
        params = dict(targets=['target.png'], anchors=['a.png', 'b.png'])
        generate_roma_predictions(self.source, output, predictor=StubPredictor(), **params)
        target = output / 'target.png.roma.json'
        target.write_text('{"untrusted":true}')
        modified = target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Incomplete or changed RoMa receipt'):
            generate_roma_predictions(self.source, output, resume=True,
                                      predictor=StubPredictor(), **params)
        self.assertEqual(target.read_bytes(), modified)

    def test_resume_fails_on_unrecorded_target_output(self):
        output = self.root / 'predictions'
        params = dict(targets=['target.png', 'rear.png'], anchors=['a.png', 'b.png'])
        class FailAtRear(StubPredictor):
            def predict(self, anchor, target, pixels, anchor_size):
                if target.name == 'rear.png':
                    raise RuntimeError('interrupted')
                return super().predict(anchor, target, pixels, anchor_size)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            generate_roma_predictions(self.source, output,
                                      predictor=FailAtRear(), **params)
        unrecorded = output / 'rear.png.roma.json'
        unrecorded.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Unrecorded RoMa output exists'):
            generate_roma_predictions(self.source, output, resume=True,
                                      predictor=StubPredictor(), **params)
        self.assertEqual(unrecorded.read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
