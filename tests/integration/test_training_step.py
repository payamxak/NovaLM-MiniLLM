"""Spec acceptance wrapper for `test_mtp_moe_and_mla_are_integrated_and_trainable`."""
import unittest

from tests.unit.test_research import test_mtp_moe_and_mla_are_integrated_and_trainable


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_mtp_moe_and_mla_are_integrated_and_trainable()
