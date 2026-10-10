"""Spec acceptance wrapper for `test_moe_shapes_routing_and_gradients`."""
import unittest

from tests.unit.test_research import test_moe_shapes_routing_and_gradients


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_moe_shapes_routing_and_gradients()
