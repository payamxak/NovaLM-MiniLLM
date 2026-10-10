"""Spec acceptance wrapper for `test_training_cli_validation_and_activation_checkpointing`."""
import unittest

from tests.integration.test_pipeline import (
    test_training_cli_validation_and_activation_checkpointing,
)


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_training_cli_validation_and_activation_checkpointing()
