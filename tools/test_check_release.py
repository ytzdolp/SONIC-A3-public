"""Release-boundary regression tests; all credential-like fixtures are synthetic."""
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("release_checker", Path(__file__).with_name("check_release.py"))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

class ReleaseChecks(unittest.TestCase):
    def test_credential_variants(self):
        fixtures = [
            ("github_" + "pat_" + "X" * 40).encode(),
            ("sk-" + "proj-" + "X" * 40).encode(),
            ("AS" + "IA" + "X" * 16).encode(),
            ("-----BEGIN " + "OPENSSH PRIVATE KEY-----").encode(),
        ]
        for value in fixtures:
            with self.subTest(kind=value[:8]):
                self.assertTrue(any(p.search(value) for p in CHECK.SECRET_PATTERNS.values()))

    def test_unicode_home_paths_and_placeholders(self):
        fixtures = ["/" + "home/" + "测试/run", "C:" + "\\" + "Users" + "\\" + "测试" + "\\" + "run"]
        for value in fixtures:
            self.assertTrue(any(p.search(value.encode()) for p in CHECK.SECRET_PATTERNS.values()))
        for value in ["/" + "home/<USER>/run", "C:" + "\\" + "Users" + "\\" + "<USER>" + "\\" + "run"]:
            self.assertFalse(any(p.search(value.encode()) for p in CHECK.SECRET_PATTERNS.values()))

    def test_large_file_is_scanned_and_values_are_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "payload.bin"
            marker = b"synthetic-sensitive-marker"
            path.write_bytes(b"x" * (6 * 1024 * 1024) + marker)
            with patch.object(CHECK, "ROOT", root):
                errors = CHECK.inspect_file(path, {"local rule #1": re.compile(marker)})
            self.assertTrue(errors)
            self.assertNotIn(marker.decode(), "\n".join(errors))

    def test_local_rules_are_ignored_but_force_added_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text(".release-check.local\n")
            private = root / ".release-check.local"
            private.write_text("private-label=synthetic-marker\n")
            subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
            with patch.object(CHECK, "ROOT", root), patch.object(CHECK, "LOCAL_RULES", private):
                self.assertNotIn(private, CHECK.git_files())
                subprocess.run(["git", "-C", str(root), "add", "-f", private.name], check=True)
                self.assertIn(private, CHECK.git_files())
                errors = CHECK.inspect_file(private, CHECK.release_patterns())
                self.assertTrue(errors)
                self.assertNotIn("synthetic-marker", "\n".join(errors))

    def test_local_rule_label_is_not_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / ".release-check.local"
            private.write_text("sensitive-label=synthetic-marker\n")
            with patch.object(CHECK, "LOCAL_RULES", private):
                patterns = CHECK.release_patterns()
            self.assertIn("local rule #1", patterns)
            self.assertFalse(any("sensitive-label" in label for label in patterns))

    def test_checker_does_not_match_itself(self):
        data = Path(__file__).with_name("check_release.py").read_bytes()
        self.assertFalse([name for name, pattern in CHECK.SECRET_PATTERNS.items() if pattern.search(data)])

if __name__ == "__main__":
    unittest.main()
