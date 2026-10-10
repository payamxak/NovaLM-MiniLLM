"""Spec acceptance wrapper for `test_mtp_horizon_loss_and_gradients`."""
import unittest

from tests.unit.test_research import test_mtp_horizon_loss_and_gradients


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_mtp_horizon_loss_and_gradients()
