import tempfile
import unittest
from pathlib import Path

from kumiki_blender.export_audit import ExportAuditError, audit_export_files


class ExportAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "source.blend").write_bytes(b"BLENDER-v300")
        (self.root / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"still")

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
