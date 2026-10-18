"""Public runtime API tests. No Factorio server is contacted by these tests.

Skip when optional runtime dependencies are absent (CI also runs static tests).
"""
import importlib.util
import unittest

HAS_FLE = importlib.util.find_spec('fle') is not None
HAS_GYM = importlib.util.find_spec('gymnasium') is not None


@unittest.skipUnless(HAS_FLE and HAS_GYM, 'Installed FLE and Gymnasium runtime required')
class RuntimeAPITests(unittest.TestCase):
    def test_action_registry_consistency(self):
        from fle_gym import actions, action_names
        catalog = actions()
        names = action_names()
        self.assertEqual(len(catalog), 124)
        self.assertEqual(len(names), 124)
        self.assertEqual(len(set(names)), 124)
        self.assertEqual(tuple(a['name'] for a in catalog), names)

    def test_unknown_scenario_rejected_before_rcon(self):
        from unittest.mock import patch
        from fle_gym import make_env
        with patch('fle_gym.api.make_env_factory') as factory:
            with self.assertRaisesRegex(ValueError, 'Unsupported Factorio scenario'):
                make_env(scenario='invalid_scenario')
            factory.assert_not_called()

if __name__ == '__main__':
    unittest.main()
