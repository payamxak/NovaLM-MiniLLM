"""Spec acceptance wrapper for `test_swiglu_shape_and_gradients`."""
import unittest

from tests.unit.test_model import test_swiglu_shape_and_gradients


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_swiglu_shape_and_gradients()
