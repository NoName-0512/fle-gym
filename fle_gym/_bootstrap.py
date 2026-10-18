from __future__ import annotations

"""Compatibility bootstrap for the validated FLE transition implementation.

The final public package is namespaced as ``fle_gym``.  The validated research
implementation predates that packaging and uses absolute intra-module imports
(e.g. ``from factorio_transition_env import ...``).  Rather than rewrite those
battle-tested modules, this bootstrap adds the private compatibility directory
only for the current interpreter.
"""

from pathlib import Path
import sys


def activate_internal_path() -> Path:
    path = Path(__file__).resolve().parent / "_internal"
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path


__all__ = ["activate_internal_path"]
