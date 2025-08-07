import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maincode.train_antibody_antigen import train, evaluate
from deepair.datasets.antibody_antigen import load_antibody_antigen_dataset, batch_generator


def _create_dummy_csv(path: str) -> None:
    rows = [
        {"antibody": "AC", "antigen": "DE", "label": "1"},
        {"antibody": "FG", "antigen": "HI", "label": "0"},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["antibody", "antigen", "label"])
        writer.writeheader()
        writer.writerows(rows)


class AntibodyAntigenTest(unittest.TestCase):
    def test_training_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_csv = os.path.join(tmpdir, "train.csv")
            model_path = os.path.join(tmpdir, "model.txt")
            _create_dummy_csv(train_csv)
            train(train_csv, epochs=1, batch_size=1, max_len=4, model_path=model_path)
            acc = evaluate(model_path, train_csv, batch_size=1, max_len=4)
            self.assertTrue(0.0 <= acc <= 1.0)
            antibodies, antigens, labels = load_antibody_antigen_dataset(train_csv)
            gen = batch_generator(antibodies, antigens, labels, batch_size=1, shuffle=False)
            ab, ag, lb = next(gen)
            self.assertEqual(len(ab[0]), 2)
            self.assertEqual(len(ag[0]), 2)
            self.assertIn(lb[0], (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
