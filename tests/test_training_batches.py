"""Ensure the notebook does not drop the last batch or request 0 training steps."""
import ast
import json
from pathlib import Path
import types
import unittest


class TrainingBatchTests(unittest.TestCase):
    def fit_parameters(self, sample_count):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "cifar10-image-classification.ipynb").read_text())
        function = next(node for cell in notebook["cells"] if cell["cell_type"] == "code"
                        and "def train_model" in "".join(cell["source"])
                        for node in ast.parse("".join(cell["source"])).body
                        if isinstance(node, ast.FunctionDef) and node.name == "train_model")
        captured = {}
        class Model:
            def fit(self, iterator, **kwargs):
                captured.update(kwargs)
                return object()
        class Iterator:
            def __init__(self, batch_size): self.batch_size = batch_size
            def __len__(self): return (sample_count + self.batch_size - 1) // self.batch_size
        namespace = {
            "train_data": (range(sample_count), range(sample_count)),
            "val_data": ([0], [0]), "get_model": Model,
            "data_gen": types.SimpleNamespace(flow=lambda x, y, batch_size: Iterator(batch_size)),
            "EarlyStopping": lambda **kwargs: object(),
            "ReduceLROnPlateau": lambda **kwargs: object(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)
        namespace["train_model"]()
        return captured

    def test_partial_batch_is_included(self):
        self.assertEqual(self.fit_parameters(129)["steps_per_epoch"], 2)

    def test_dataset_smaller_than_batch_size_gets_a_training_step(self):
        self.assertEqual(self.fit_parameters(3)["steps_per_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
