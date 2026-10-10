"""Spec acceptance wrapper for `test_clean_quality_deduplicate_pack_and_shard`."""
import unittest

from tests.integration.test_pipeline import test_clean_quality_deduplicate_pack_and_shard


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_clean_quality_deduplicate_pack_and_shard()
