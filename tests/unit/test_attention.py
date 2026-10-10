"""Spec acceptance wrapper for `test_sdpa_matches_reference`."""
import unittest

from tests.unit.test_model import test_sdpa_matches_reference


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_sdpa_matches_reference()
