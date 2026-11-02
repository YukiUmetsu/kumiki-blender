"""Anatomical head graft preservation tests; synthetic, not monument likeness."""
import unittest
import numpy as np
import trimesh
from kumiki_blender.head_graft import graft_anatomical_head


def sphere(radius=1., center=(0,0,0), subdivisions=5):
    mesh=trimesh.creation.icosphere(subdivisions=subdivisions,radius=radius)
    mesh.apply_translation(center)
    return mesh


class HeadGraftTests(unittest.TestCase):
    def make_scenes(self):
        base=trimesh.Scene()
        base.add_geometry(sphere(1.8,(0,36.,.5),subdivisions=3),
                          geom_name="Head | integrated cheekbone jaw and cranium",
                          node_name="Head | integrated cheekbone jaw and cranium")
        base.add_geometry(sphere(3.,(0,22.,0),subdivisions=2),
                          geom_name="Statue | robe",node_name="Statue | robe")
        base.add_geometry(sphere(.3,(0,36.,2.),subdivisions=2),
                          geom_name="Face | carved closed mouth",
                          node_name="Face | carved closed mouth")
        base.add_geometry(sphere(1.9,(0,36.,-.2),subdivisions=2),
                          geom_name="Hair | mantle",node_name="Hair | mantle")
        donor=trimesh.Scene()
        donor.add_geometry(sphere(1.75,(0,36.,.55),subdivisions=6),
                           geom_name="Head | anatomical quad-topology clay sculpt",
                           node_name="Head | anatomical quad-topology clay sculpt")
        return base,donor

    def test_world_head_swap_preserves_body_and_hair(self):
        base,donor=self.make_scenes()
        result,audit=graft_anatomical_head(base,donor)
        self.assertEqual(audit["status"],"HOLD_photo_likeness_not_verified")
        self.assertGreater(audit["donor_head_vertices"],10000)
        self.assertIn("Statue | robe",result.graph.nodes_geometry)
        self.assertIn("Hair | mantle",result.graph.nodes_geometry)
        self.assertNotIn("Head | integrated cheekbone jaw and cranium",result.graph.nodes_geometry)
        self.assertNotIn("Face | carved closed mouth",result.graph.nodes_geometry)
        self.assertIn("Head | integrated cheekbone jaw and cranium",base.graph.nodes_geometry)
        self.assertEqual(audit["preserved_nodes"],2)

    def test_bound_fit_scales_generic_donor(self):
        base,donor=self.make_scenes()
        donor.delete_geometry("Head | anatomical quad-topology clay sculpt")
        donor.add_geometry(sphere(.35,(.4,8.,.2),subdivisions=6),
                           geom_name="Head | anatomical quad-topology clay sculpt",
                           node_name="Head | anatomical quad-topology clay sculpt")
        result,audit=graft_anatomical_head(base,donor,alignment="bounds")
        self.assertGreater(audit["alignment"]["uniform_scale"],1)
        self.assertIn("Head | anatomical quad-topology HOLD",result.graph.nodes_geometry)

    def test_wrong_height_rejected(self):
        base,donor=self.make_scenes()
        node="Head | anatomical quad-topology clay sculpt"
        donor.graph.update(frame_to=node,
            matrix=trimesh.transformations.translation_matrix((0,20,0)))
        with self.assertRaisesRegex(ValueError,"height"):
            graft_anatomical_head(base,donor)

    def test_donor_cannot_be_tiny_placeholder(self):
        base,donor=self.make_scenes()
        donor.delete_geometry("Head | anatomical quad-topology clay sculpt")
        donor.add_geometry(sphere(1.75,(0,36.,.55),subdivisions=2),
                           geom_name="Head | anatomical quad-topology clay sculpt",
                           node_name="Head | anatomical quad-topology clay sculpt")
        with self.assertRaisesRegex(ValueError,"dense"):
            graft_anatomical_head(base,donor)

    def test_replaces_hair_only_when_explicitly_requested(self):
        base,donor=self.make_scenes()
        donor.add_geometry(sphere(.7,(1.5,35.,-.2),subdivisions=2),
                           geom_name="Hair | sculpted temple",
                           node_name="Hair | sculpted temple")
        result,audit=graft_anatomical_head(base,donor,replace_hair=True)
        self.assertNotIn("Hair | mantle",result.graph.nodes_geometry)
        self.assertIn("Study | Hair | sculpted temple",result.graph.nodes_geometry)
        self.assertEqual(audit["donor_hair_meshes"],1)

    def test_transfers_soapstone_material_with_metric_uv(self):
        base,donor=self.make_scenes()
        name="Head | integrated cheekbone jaw and cranium"
        material=trimesh.visual.material.PBRMaterial(
            name="Test | tessellated soapstone",baseColorFactor=[.67,.67,.65,1.])
        original=base.geometry[name]
        original.visual=trimesh.visual.TextureVisuals(
            uv=np.zeros((len(original.vertices),2)),material=material)
        result,audit=graft_anatomical_head(base,donor)
        head=result.geometry["Head | anatomical quad-topology HOLD"]
        self.assertEqual(head.visual.material.name,material.name)
        self.assertGreater(np.ptp(head.visual.uv[:,0]),1.)
        self.assertGreater(np.ptp(head.visual.uv[:,1]),1.)
        self.assertIn("metric",audit["head_uv_projection"])

    def test_world_transform_is_respected(self):
        base,donor=self.make_scenes()
        node="Head | anatomical quad-topology clay sculpt"
        donor.graph.update(frame_to=node,
            matrix=trimesh.transformations.translation_matrix((.18,0,-.12)))
        result,audit=graft_anatomical_head(base,donor)
        self.assertEqual(audit["alignment"]["mode"],"world")
        self.assertAlmostEqual(audit["head_bounds_world_m"][0][0],-1.57,delta=.05)

    def test_missing_hair_rejected(self):
        base,donor=self.make_scenes()
        with self.assertRaisesRegex(ValueError,"no hair"):
            graft_anatomical_head(base,donor,replace_hair=True)


if __name__=="__main__":
    unittest.main()
