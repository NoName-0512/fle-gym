#!/usr/bin/env python3
"""Checks the public import without requiring an FLE server."""
from fle_gym import FactorioLabEnv
print("PASS: import from fle_gym import FactorioLabEnv")
print("PASS: environment class:", FactorioLabEnv)
print("Note: live reset/step still needs an FLE/Factorio instance")
