"""Render-pixel ray tests require no Blender installation."""

import math
import unittest

from kumiki_blender.render_ray import pixel_camera_ray


class RenderRayTests(unittest.TestCase):
    FRAME = [(-3.2,-3.2,-1),(3.2,-3.2,-1),
             (3.2,3.2,-1),(-3.2,3.2,-1)]

    def test_ortho_front_center_and_edges(self):
        origin, direction = pixel_camera_ray((350,350),(700,700),self.FRAME,"ORTHO")
        self.assertEqual(origin,(0.,0.,0.))
        self.assertEqual(direction,(0.,0.,-1.))
        self.assertEqual(pixel_camera_ray((0,0),(700,700),self.FRAME,"ORTHO")[0],(-3.2,3.2,0.))
        self.assertEqual(pixel_camera_ray((700,700),(700,700),self.FRAME,"ORTHO")[0],(3.2,-3.2,0.))

    def test_perspective_center_and_off_axis(self):
        origin, direction = pixel_camera_ray((500,250),(1000,500),
            [(-2,-1,-2),(2,-1,-2),(2,1,-2),(-2,1,-2)],"PERSP")
        self.assertEqual(origin,(0.,0.,0.))
        self.assertEqual(direction,(0.,0.,-1.))
        _, ray = pixel_camera_ray((750,125),(1000,500),
            [(-2,-1,-2),(2,-1,-2),(2,1,-2),(-2,1,-2)],"PERSP")
        self.assertAlmostEqual(math.sqrt(sum(v*v for v in ray)),1.)
        self.assertGreater(ray[0],0)
        self.assertGreater(ray[1],0)
        self.assertLess(ray[2],0)

    def test_shifted_camera_and_non_square_output(self):
        frame=[(-1,-2,-1),(3,-2,-1),(3,2,-1),(-1,2,-1)]
        origin,_=pixel_camera_ray((200,100),(800,400),frame,"ORTHO")
        self.assertEqual(origin,(0.,1.,0.))
        origin,_=pixel_camera_ray((400,200),(800,400),frame,"ORTHO")
        self.assertEqual(origin,(1.,0.,0.))

    def test_rejects_invalid_pixel_and_frame(self):
        for pixel in ((-1,10),(701,10),(math.nan,0)):
            with self.subTest(pixel=pixel),self.assertRaises(ValueError):
                pixel_camera_ray(pixel,(700,700),self.FRAME,"ORTHO")
        with self.assertRaisesRegex(ValueError,"Only orthographic"):
            pixel_camera_ray((0,0),(700,700),self.FRAME,"PANO")
        with self.assertRaisesRegex(ValueError,"Invalid planar"):
            pixel_camera_ray((0,0),(700,700),[(0,0,0)]*4,"ORTHO")
        with self.assertRaisesRegex(ValueError,"Perspective frame"):
            pixel_camera_ray((0,0),(700,700),[(x,y,1) for x,y,_ in self.FRAME],"PERSP")


if __name__ == "__main__":
    unittest.main()
