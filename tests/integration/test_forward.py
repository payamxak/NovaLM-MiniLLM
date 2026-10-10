"""Spec acceptance wrapper for `test_model_forward_loss_backward_and_weight_tying`."""
import unittest

from tests.unit.test_model import test_model_forward_loss_backward_and_weight_tying


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_model_forward_loss_backward_and_weight_tying()
