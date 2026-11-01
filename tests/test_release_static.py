"""Static release tests: Python standard library only, no Factorio server."""
from pathlib import Path
import ast
import re
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


def version_from_init():
    module = ast.parse((ROOT / 'fle_gym/__init__.py').read_text(encoding='utf8'))
    for stmt in module.body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in stmt.targets):
            return ast.literal_eval(stmt.value)
    raise AssertionError('missing __version__')


class ReleaseStaticTests(unittest.TestCase):
    def test_version_matches_build_metadata(self):
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf8'))
        self.assertEqual(config['project']['version'], version_from_init())
        self.assertRegex(version_from_init(), r'^[0-9]+\.[0-9]+\.[0-9]+(?:rc[0-9]+)?$')

    def test_telemetry_keys_are_distinct(self):
        internal = ROOT / 'fle_gym/_internal'
        contents = '\n'.join(p.read_text(encoding='utf8') for p in internal.glob('*.py'))
        self.assertNotIn('runtime_action_runtime', contents)
        for key in ('runtime_action_inventory', 'runtime_action_factory', 'runtime_action_extended'):
            with self.subTest(key=key):
                self.assertIn(key, contents)

    def test_scenario_validation_is_explicit(self):
        source = (ROOT / 'fle_gym/api.py').read_text(encoding='utf8')
        self.assertIn('Unsupported Factorio scenario', source)
        self.assertIn('default_lab_scenario', source)
        self.assertIn('open_world', source)

    def test_generated_artifacts_ignored(self):
        text = (ROOT / '.gitignore').read_text(encoding='utf8')
        for value in ('world_saves/', 'test_logs/', '*.zip', '.env'):
            with self.subTest(value=value):
                self.assertIn(value, text)

    def test_all_python_files_parse(self):
        for path in (ROOT / 'fle_gym').rglob('*.py'):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding='utf8'), filename=str(path))

if __name__ == '__main__':
    unittest.main()
