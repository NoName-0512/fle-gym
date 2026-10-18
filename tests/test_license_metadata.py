"""License metadata checks; requires Python 3.11+ (tomllib)."""
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LicenseTests(unittest.TestCase):
    def test_project_is_noncommercial(self):
        meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))
        self.assertEqual(meta["project"]["license"], "LicenseRef-FLE-Gym-NonCommercial-1.0")
        self.assertIn("LICENSE", meta["project"]["license-files"])
        self.assertIn("THIRD_PARTY_NOTICES.md", meta["project"]["license-files"])

    def test_personal_copyright_and_upstream_attribution(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf8")
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf8")
        readme = (ROOT / "README.md").read_text(encoding="utf8")
        self.assertTrue(license_text.startswith("FLE-Gym Personal Non-Commercial License"))
        self.assertIn("Copyright (c) 2026 Duran Reddy", license_text)
        self.assertIn("commercial use", license_text.lower())
        self.assertIn("prior written", license_text.lower())
        self.assertNotIn("Witwatersrand", license_text)
        self.assertIn("Copyright (c) 2025 Factorio Learning Environment Contributors", notices)
        self.assertIn("JackHopkins/factorio-learning-environment", notices)
        self.assertIn("game assets and game code", notices)
        self.assertIn("Non-Commercial", readme)
        self.assertNotIn("Witwatersrand", readme)
