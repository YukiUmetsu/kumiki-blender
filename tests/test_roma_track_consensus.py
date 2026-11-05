import unittest
from kumiki_blender.reconstruction.roma_track_consensus import agree_track_predictions


class TrackConsensusTests(unittest.TestCase):
    def test_requires_independent_agreeing_anchors_and_subject(self):
        def entry(track, anchor, x, subject=True):
            return {'track_id':track,'anchor':anchor,'pixel':[x,0],'xyz':[1,2,3], 'subject_candidate':subject}
        predictions=[entry(1,'a',10),entry(1,'b',12),entry(2,'a',1),entry(2,'b',9),
                     entry(3,'a',1),entry(3,'a',1),entry(4,'a',1),entry(4,'b',1,False),
                     entry(5,'a',1),entry(5,'b',2),entry(5,'c',20)]
        result=agree_track_predictions(predictions)
        self.assertEqual([p['track_id'] for p in result],[1])
        self.assertEqual(result[0]['pixel'],[11,0])
        self.assertEqual(result[0]['anchors'],['a','b'])

    def test_invalid_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            agree_track_predictions([],max_error_pixels=0)
