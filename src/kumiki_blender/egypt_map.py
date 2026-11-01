"""Backward-compatible geographic API for the categorized Egypt atlas package.

The implementation now lives under ``geography.egypt.terrain``; historic
imports and recipe validation continue to use this stable path.
"""

from .geography.egypt.terrain import *  # noqa: F401,F403
