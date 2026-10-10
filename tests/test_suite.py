import runpy
import unittest
from pathlib import Path


class ProjectChecks(unittest.TestCase):
    def test_model_data_tokenizer_training_inference_and_research(self):
        root = Path(__file__).resolve().parents[1]
        count = 0
        for relative in ("unit/test_model.py", "integration/test_pipeline.py",
                         "unit/test_research.py"):
            namespace = runpy.run_path(str(root / "tests" / relative))
            for name, test in namespace.items():
                if name.startswith("test_") and callable(test):
                    with self.subTest(test=name):
                        test()
                        count += 1
        self.assertEqual(count, 26)
