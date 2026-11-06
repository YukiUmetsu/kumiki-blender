import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from kumiki_blender.reconstruction.sfm_pair_diagnostics import diagnose_pairs, write_pair_visual


class PairDiagnosticsTests(unittest.TestCase):
    def test_matching_stages_and_track_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); db = root / 'db.sqlite'; points = root / 'points3D.txt'
            points.write_text('7 0 0 0 255 255 255 0.1 1 0\n')
            with sqlite3.connect(db) as conn:
                conn.executescript('CREATE TABLE images(image_id INTEGER, name TEXT);'
                                   'CREATE TABLE matches(pair_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);'
                                   'CREATE TABLE two_view_geometries(pair_id INTEGER,rows INTEGER,cols INTEGER,data BLOB,config INTEGER);'
                                   'CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);')
                conn.executemany('INSERT INTO images VALUES(?,?)', [(1,'anchor.png'),(2,'ok.png'),(3,'failed.png'),(4,'empty.png'),(5,'unknown.png')])
                conn.executemany('INSERT INTO keypoints VALUES(?,?,?,?)', [(i, 1, 2, struct.pack('<ff', 12, 12)) for i in range(1, 6)])
                pair = lambda b: 2147483647 + b
                blob = struct.pack('<II', 0, 0)
                conn.executemany('INSERT INTO matches VALUES(?,?,?,?)', [(pair(2),1,2,blob),(pair(3),1,2,blob),(pair(4),0,2,b'')])
                conn.execute('INSERT INTO two_view_geometries VALUES(?,?,?,?,?)', (pair(2),1,2,blob,2))
                conn.execute('INSERT INTO two_view_geometries VALUES(?,?,?,?,?)', (pair(3),0,2,b'',0))
            for name in ('anchor.png','ok.png','failed.png','empty.png','unknown.png'):
                Image.new('RGB',(32,32)).save(root/name)
            records = diagnose_pairs(db, points, {'anchor.png'}, {'ok.png','failed.png','empty.png','unknown.png'})
            by_target = {r['target']:r for r in records}
            self.assertEqual(by_target['ok.png']['anchor_observations_on_3d_tracks'], 1)
            self.assertEqual(by_target['failed.png']['stage'], 'geometric-verification-rejected')
            self.assertEqual(by_target['empty.png']['stage'], 'no-tentative-matches')
            self.assertEqual(by_target['unknown.png']['stage'], 'no-match-record / attempted-unknown')
            output=root/'pair.png'; write_pair_visual(db,root,points,by_target['ok.png'],output)
            self.assertTrue(output.is_file())
