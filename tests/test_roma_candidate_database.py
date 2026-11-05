import hashlib
import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from kumiki_blender.reconstruction.roma_candidate_database import prepare_roma_candidate_database


class CandidateDatabaseTests(unittest.TestCase):
    def test_import_is_isolated_and_requires_real_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'source'; (source/'sparse/0').mkdir(parents=True)
            (source/'text-models/0').mkdir(parents=True); (source/'images').mkdir()
            names=['a.png','b.png','c.png','target.png']; entries=[]
            for name in names:
                image=source/'images'/name; image.write_bytes(name.encode());entries.append({'filename':name,'sha256':hashlib.sha256(image.read_bytes()).hexdigest()})
            (source/'staged-reference-index.json').write_text(json.dumps({'images':entries}))
            (source/'text-models/0/images.txt').write_text(''.join(f'{i} 1 0 0 0 0 0 0 1 {name}\n0 0 -1\n' for i,name in enumerate(names[:3],1)))
            (source/'text-models/0/points3D.txt').write_text('10 1 2 3 255 255 255 0.1 1 0 2 0\n')
            with sqlite3.connect(source/'database.db') as db:
                db.executescript('CREATE TABLE images(image_id INTEGER,name TEXT,camera_id INTEGER); CREATE TABLE cameras(camera_id INTEGER,width INTEGER,height INTEGER); CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB); CREATE TABLE descriptors(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB); CREATE TABLE matches(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB); CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB,config INTEGER);')
                db.executemany('INSERT INTO images VALUES(?,?,1)',[(i,name) for i,name in enumerate(names,1)]);db.execute('INSERT INTO cameras VALUES(1,100,100)');db.execute('INSERT INTO keypoints VALUES(4,1,4,?)',(struct.pack('<ffff',1,1,1,0),));db.execute('INSERT INTO descriptors VALUES(4,1,2,?)',(b'12',))
            predictions=[{'track_id':10,'anchor':a,'xyz':[1,2,3],'pixel':[20+i,30],'subject_candidate':True} for i,a in enumerate(names[:2])]
            evidence=root/'predictions.json';evidence.write_text(json.dumps({'target':'target.png','predictions':predictions}))
            target=root/'out';report=prepare_roma_candidate_database(source,evidence,target)
            self.assertEqual(report['candidate_tracks'],1)
            with sqlite3.connect(source/'database.db') as db:self.assertEqual(db.execute('SELECT rows FROM keypoints').fetchone()[0],1)
            with sqlite3.connect(target/'database.db') as db:
                self.assertEqual(db.execute('SELECT rows FROM keypoints').fetchone()[0],2)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM matches').fetchone()[0],2)
            with self.assertRaises(ValueError):prepare_roma_candidate_database(source,evidence,target)
            predictions[1]['anchor']='c.png';evidence.write_text(json.dumps({'target':'target.png','predictions':predictions}))
            with self.assertRaisesRegex(ValueError,'exactly one observation'):
                prepare_roma_candidate_database(source,evidence,root/'invalid')
            self.assertFalse((root/'invalid').exists())
