"""Spec acceptance wrapper for `test_sentencepiece_training_roundtrip_and_chat_template`."""
import unittest

from tests.integration.test_pipeline import test_sentencepiece_training_roundtrip_and_chat_template


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_sentencepiece_training_roundtrip_and_chat_template()
