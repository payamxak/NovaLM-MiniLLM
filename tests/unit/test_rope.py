"""Spec acceptance wrapper for `test_rope_extends_cache_and_preserves_shape`."""
import unittest

from tests.unit.test_model import test_rope_extends_cache_and_preserves_shape


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_rope_extends_cache_and_preserves_shape()
