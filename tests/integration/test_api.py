"""Spec acceptance wrapper for `test_fastapi_health_info_completion_and_chat`."""
import unittest

from tests.integration.test_pipeline import test_fastapi_health_info_completion_and_chat


class AcceptanceCheck(unittest.TestCase):
    def test_acceptance(self):
        test_fastapi_health_info_completion_and_chat()
