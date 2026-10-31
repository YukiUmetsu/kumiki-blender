import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kumiki_blender.blender_lighting import (
    add_architectural_reflections, plan_architectural_reflections,
)


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

    def test_reflection_cards_link_only_the_supplied_glass_receivers(self):
        receiver = SimpleNamespace(type="MESH")
        linked = SimpleNamespace(objects=SimpleNamespace(link=Mock()))
        stage = SimpleNamespace(objects=SimpleNamespace(link=Mock()))
        created_objects = []

        def make_object(name, light):
            obj = SimpleNamespace(name=name, location=None,
                                  light_linking=SimpleNamespace(receiver_collection=None))
            created_objects.append(obj)
            return obj

        bpy = SimpleNamespace(data=SimpleNamespace(
            collections=SimpleNamespace(new=Mock(return_value=linked)),
            lights=SimpleNamespace(new=Mock(side_effect=lambda name, type:
                                            SimpleNamespace(name=name, type=type))),
            objects=SimpleNamespace(new=Mock(side_effect=make_object)),
        ))
        with patch("kumiki_blender.blender_lighting.aim_at"):
            lights = add_architectural_reflections(
                bpy, stage, center_x=6, facade_y=-3.6, center_z=5, span=12,
                glass_receivers=(receiver,))
        self.assertEqual(len(lights), 2)
        linked.objects.link.assert_called_once_with(receiver)
        self.assertEqual(stage.objects.link.call_count, 2)
        self.assertTrue(all(obj.light_linking.receiver_collection is linked
                            for obj in created_objects))


if __name__ == '__main__':
    unittest.main()
