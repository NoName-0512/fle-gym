"""Release regression checks for the internal-module naming refactor (stdlib)."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ROOT / "fle_gym" / "_internal"


class InternalNamingTests(unittest.TestCase):
    def test_expected_modules_exist_and_old_modules_absent(self):
        for stem in (
            "factorio_openended_env_feasibility",
            "factorio_openended_env_extended",
        ):
            with self.subTest(module=stem):
                self.assertTrue((INTERNAL / (stem + ".py")).is_file())
        for stem in (
            "factorio_openended_env_research_ajweuzp",
            "factorio_openended_env_research_yqeibtv",
        ):
            with self.subTest(module=stem):
                self.assertFalse((INTERNAL / (stem + ".py")).exists())

    def test_no_legacy_suffixes_in_package_source(self):
        for path in (ROOT / "fle_gym").rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotIn("ajweuzp", content.lower())
                self.assertNotIn("yqeibtv", content.lower())

    def test_renamed_classes_and_imports_present(self):
        feasibility = (INTERNAL / "factorio_openended_env_feasibility.py").read_text(encoding="utf-8")
        extended = (INTERNAL / "factorio_openended_env_extended.py").read_text(encoding="utf-8")
        machine = (INTERNAL / "factorio_openended_env_machine.py").read_text(encoding="utf-8")
        self.assertIn("class FactorioOpenEndedEnvFeasibility(", feasibility)
        self.assertIn("class FactorioOpenEndedEnvExtended(", extended)
        self.assertIn("FactorioOpenEndedEnvExtended", machine)
        for path in (ROOT / "fle_gym").rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_all_internal_absolute_imports_resolve(self):
        """Catch misspelled internal imports even without runtime dependencies."""
        for path in (ROOT / "fle_gym").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                    continue
                name = node.module
                if name.startswith("factorio_openended_env_") or name in {
                    "factorio_transition_env", "env_factory", "environment_contract",
                }:
                    with self.subTest(source=str(path.relative_to(ROOT)), module=name):
                        self.assertTrue(
                            (INTERNAL / (name + ".py")).is_file(),
                            f"Missing internal module: {name}",
                        )
