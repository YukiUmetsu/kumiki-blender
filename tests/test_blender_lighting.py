import unittest

from kumiki_blender.blender_lighting import plan_architectural_reflections


class ReflectionRigTests(unittest.TestCase):
    def test_plans_distinct_sky_softboxes_without_blender(self):
        cards = plan_architectural_reflections(center_x=6, facade_y=-3.6,
                                               center_z=5, span=12.4)
        self.assertEqual(len(cards), 2)
        self.assertTrue(all(card.width > 0 and card.height > card.width for card in cards))
        self.assertTrue(all(card.location[1] < -3.6 for card in cards))
        self.assertNotEqual(cards[0].location, cards[1].location)
        self.assertTrue(all(card.energy > 0 for card in cards))
        self.assertEqual(cards, plan_architectural_reflections(center_x=6, facade_y=-3.6,
                                                               center_z=5, span=12.4))

    def test_rejects_nonfinite_or_excessive_reflection_rigs(self):
        for span in (0, 1.9, 121, float('nan'), True):
            with self.subTest(span=span), self.assertRaises(ValueError):
                plan_architectural_reflections(center_x=0, facade_y=0, center_z=5, span=span)


if __name__ == '__main__':
    unittest.main()
