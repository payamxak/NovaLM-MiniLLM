"""Spec acceptance wrapper for `test_attention_backends_and_mha_shapes`."""
import unittest

from tests.unit.test_components import ComponentContracts


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        ComponentContracts().test_attention_backends_and_mha_shapes()
