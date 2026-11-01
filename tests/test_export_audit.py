import tempfile
import unittest
from pathlib import Path
import struct
import zlib

from kumiki_blender.export_audit import ExportAuditError, audit_export_files


def _one_pixel_png() -> bytes:
    def chunk(tag: bytes, content: bytes) -> bytes:
        return (struct.pack(">I",len(content))+tag+content+
                struct.pack(">I",zlib.crc32(tag+content)))
    return (b"\x89PNG\r\n\x1a\n"+
            chunk(b"IHDR",struct.pack(">IIBBBBB",1,1,8,2,0,0,0))+
            chunk(b"IDAT",zlib.compress(b"\x00\xff\x00\x00"))+
            chunk(b"IEND",b""))


class ExportAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "source.blend").write_bytes(b"BLENDER-v300")
        (self.root / "preview.png").write_bytes(_one_pixel_png())

    def test_readable_editable_source_and_preview_are_reported(self):
        report = audit_export_files(self.root, ["source.blend", "preview.png"])
        self.assertGreater(report["source.blend"], 7)
        self.assertGreater(report["preview.png"], 8)

    def test_missing_preview_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ExportAuditError, "missing"):
            audit_export_files(self.root, ["previews/detail.png"])

    def test_mislabeled_files_fail_signature_validation(self):
        (self.root / "preview.png").write_bytes(b"not-a-png")
        with self.assertRaisesRegex(ExportAuditError, "PNG"):
            audit_export_files(self.root, ["preview.png"])

    def test_truncated_or_corrupt_render_fails_even_with_correct_png_signature(self):
        path=self.root/"preview.png"
        original=path.read_bytes()
        path.write_bytes(original[:-8])
        with self.assertRaisesRegex(ExportAuditError,"Truncated PNG"):
            audit_export_files(self.root,["preview.png"])
        path.write_bytes(original[:-14]+bytes([original[-14]^1])+original[-13:])
        with self.assertRaisesRegex(ExportAuditError,"PNG checksum"):
            audit_export_files(self.root,["preview.png"])

    def test_relative_path_escape_is_rejected(self):
        with self.assertRaisesRegex(ExportAuditError, "relative"):
            audit_export_files(self.root, ["../other.blend"])

    def test_symlinked_parent_cannot_escape_artifact_root(self):
        with tempfile.TemporaryDirectory() as external:
            (Path(external) / "detail.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"outside")
            (self.root / "previews").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ExportAuditError, "root"):
                audit_export_files(self.root, ["previews/detail.png"])


if __name__ == "__main__":
    unittest.main()
