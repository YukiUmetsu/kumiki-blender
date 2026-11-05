from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kumiki_blender.document_face import DocumentFaceError, render_document_face

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

FONT = "/System/Library/Fonts/Supplemental/Georgia.ttf"


@unittest.skipUnless(Image and Path(FONT).exists(), "needs Pillow and the test font")
class DocumentFace(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = {"schema_version": 1, "width": 400, "height": 300, "fonts": {"body": FONT},
                     "blocks": [{"kind": "text", "text": "HELLO", "size": 40}, {"kind": "rule"},
                                {"kind": "row", "label": "Total", "value": "$120", "size": 30, "value_size": 44}]}

    def test_renders_ink_on_paper(self):
        out = self.root / "face.png"
        render_document_face(self.spec, out, root=self.root)
        image = Image.open(out).convert("L")
        self.assertEqual(image.size, (400, 300))
        self.assertLess(image.getextrema()[0], 120)

    def test_content_taller_than_the_page_fails(self):
        self.spec["height"] = 70
        with self.assertRaisesRegex(DocumentFaceError, "tall"):
            render_document_face(self.spec, self.root / "face.png", root=self.root)

    def test_unknown_block_and_font_fail(self):
        with self.assertRaisesRegex(DocumentFaceError, "Unknown document block"):
            render_document_face({**self.spec, "blocks": [{"kind": "stamp"}]}, self.root / "f.png", root=self.root)
        with self.assertRaisesRegex(DocumentFaceError, "undeclared font"):
            render_document_face({**self.spec, "fonts": {}}, self.root / "f.png", root=self.root)


if __name__ == "__main__":
    unittest.main()
