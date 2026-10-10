"""Spec acceptance wrapper for `test_kv_cache_matches_full_prefix_logits`."""
import unittest

from tests.integration.test_pipeline import test_kv_cache_matches_full_prefix_logits


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_kv_cache_matches_full_prefix_logits()
