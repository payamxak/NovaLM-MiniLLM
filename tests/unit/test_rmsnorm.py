"""Spec acceptance wrapper for `test_rmsnorm_matches_reference_and_has_gradients`."""
import unittest

from tests.unit.test_model import test_rmsnorm_matches_reference_and_has_gradients


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_rmsnorm_matches_reference_and_has_gradients()
