from pathlib import Path
import sys
import unittest
import uuid

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import main  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_missing_kaggle_dataset_falls_back_to_demo(self) -> None:
        dataset_source = main.resolve_dataset_source(dataset_arg=None, use_demo_data=False)

        self.assertTrue(dataset_source.using_demo)
        self.assertEqual(dataset_source.path, main.DEMO_DATASET)
        self.assertIsNotNone(dataset_source.fallback_reason)

    def test_cv_splits_scale_down_for_small_class_counts(self) -> None:
        labels = np.array(["Yes", "Yes", "No", "No"])

        self.assertEqual(main.get_cv_splits(labels), 2)

    def test_demo_pipeline_runs_end_to_end(self) -> None:
        dataset_source = main.resolve_dataset_source(dataset_arg=None, use_demo_data=True)
        output_path = main.OUTPUTS_DIR / f"test_preprocessed_telco_demo_{uuid.uuid4().hex}.csv"

        summary = main.run_pipeline(dataset_source, output_path)

        self.assertTrue(output_path.exists())
        self.assertGreater(summary.input_shape[0], 0)
        self.assertGreater(summary.train_shape[0], 0)
        self.assertGreaterEqual(summary.metrics.accuracy, 0.0)
        self.assertEqual(summary.dataset_source.path, main.DEMO_DATASET)


if __name__ == "__main__":
    unittest.main()
