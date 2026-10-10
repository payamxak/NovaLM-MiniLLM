"""Spec acceptance wrapper for `test_latent_attention_cache_equivalence`."""
import unittest

from tests.unit.test_research import test_latent_attention_cache_equivalence


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_latent_attention_cache_equivalence()
