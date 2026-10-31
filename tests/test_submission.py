"""Exercise the notebook's data/submission functions without TensorFlow or a dataset."""

import ast
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import pandas as pd


def notebook_functions():
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "cifar10-image-classification.ipynb").read_text()
    )
    namespace = {
        "np": np,
        "pd": pd,
        "os": os,
        "tqdm": lambda iterable, **kwargs: iterable,
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        if not source.strip() or source.lstrip().startswith("!"):
            continue
        tree = ast.parse(source)
        selected = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "load_images", "load_test_images", "create_submission_file"
            }:
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {"CLASS_LABELS", "CLASS_TO_INDEX"}
                for target in node.targets
            ):
                selected.append(node)
        if selected:
            exec(compile(ast.Module(body=selected, type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.ns = notebook_functions()
        self.ns["cv2"] = types.SimpleNamespace(
            imread=lambda path: np.full((32, 32, 3), int(Path(path).stem), dtype=np.uint8),
            resize=lambda image, size: image,
        )

    def test_training_label_order_does_not_change_class_encoding(self):
        labels = pd.DataFrame({"id": ["5.png", "0.png"], "label": ["dog", "airplane"]})
        _, encoded = self.ns["load_images"]("images", labels)
        np.testing.assert_array_equal(encoded, [5, 0])

    def test_submission_ids_remain_aligned_with_predictions(self):
        ids = [9, 2, 7]
        images = self.ns["load_test_images"]("images", ids)
        np.testing.assert_allclose(images[:, 0, 0, 0] * 255, ids)

        def predict(batch):
            return np.eye(10)[np.rint(batch[:, 0, 0, 0] * 255).astype(int)]

        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.csv"
            output = Path(directory) / "submission.csv"
            pd.DataFrame({"id": ids, "label": ["cat"] * 3}).to_csv(sample, index=False)
            self.ns["create_submission_file"](
                types.SimpleNamespace(predict=predict), images, sample, output
            )
            result = pd.read_csv(output)
            self.assertEqual(result["id"].tolist(), ids)
            self.assertEqual(result["label"].tolist(), ["truck", "bird", "horse"])

    def test_unreadable_image_fails_instead_of_silently_dropping_a_row(self):
        self.ns["cv2"].imread = lambda path: None
        with self.assertRaises(FileNotFoundError):
            self.ns["load_test_images"]("images", [1])

    def test_unknown_training_class_is_rejected(self):
        labels = pd.DataFrame({"id": ["1.png"], "label": ["unknown"]})
        with self.assertRaisesRegex(ValueError, "Unknown training label"):
            self.ns["load_images"]("images", labels)

    def test_prediction_shape_and_image_count_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.csv"
            pd.DataFrame({"id": [1, 2], "label": ["cat", "cat"]}).to_csv(sample, index=False)
            model = types.SimpleNamespace(predict=lambda data: np.zeros((2, 9)))
            with self.assertRaisesRegex(ValueError, "image count"):
                self.ns["create_submission_file"](model, np.zeros((1, 32, 32, 3)), sample)
            with self.assertRaisesRegex(ValueError, "10 class scores"):
                self.ns["create_submission_file"](model, np.zeros((2, 32, 32, 3)), sample)


if __name__ == "__main__":
    unittest.main()
