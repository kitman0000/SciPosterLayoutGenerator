import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_schema import config, document

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from sciposter_inference.model import EncoderBlock, LayoutModel, load_checkpoint
    from sciposter_inference.predictor import generate, prepare_batch
    from sciposter_inference.schema import validate_input


@unittest.skipUnless(HAS_TORCH, "Install inference dependencies to run tensor tests")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)

    def test_feature_alignment(self):
        cfg = config()
        data = document(2)
        panels = data["posters"][0]["panels"]
        panels[0]["text_len"], panels[1]["text_len"] = 2, 7
        seq, features = prepare_batch(validate_input(data, cfg), "cpu")
        self.assertEqual(seq[0, :12].tolist(), [39, 0, 41, 41, 41, 41, 1, 41, 41, 41, 41, 38])
        self.assertEqual(features[0, 0, :12].tolist(), [10] + [2] * 5 + [7] * 5 + [0])
        self.assertEqual(features[0, 4, :12].tolist(), [10] + [0] * 5 + [1] * 5 + [10])
        self.assertEqual(tuple(features.shape), (1, 5, 51))

    def test_historical_encoder_residual(self):
        block = EncoderBlock(config()).eval()
        with torch.no_grad():
            block.self_attn.out_proj.weight.zero_()
            block.self_attn.out_proj.bias.zero_()
            block.linear2.weight.zero_()
            block.linear2.bias.zero_()
        values = torch.randn(2, 4, 8) + 10
        torch.testing.assert_close(block(values), block.norm1(values))

    def test_checkpoint_roundtrip_and_strict_loading(self):
        cfg = config()
        original = LayoutModel(cfg).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            state = {"model.module." + k: v for k, v in original.state_dict().items()}
            torch.save(state, path)
            restored = load_checkpoint(path, cfg, "cpu")
            self.assertEqual(generate(original, document()), generate(restored, document()))
            del state["model.module.c_head.1.weight"]
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, cfg, "cpu")

    def test_zero_weights_decode_and_padding(self):
        model = LayoutModel(config())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        data = document(10)
        result = generate(model, data)
        for panel in result["posters"][0]["panels"]:
            self.assertTrue(panel["valid"])
            self.assertEqual(panel["bbox"], [0.0, 0.0, 1 / 32, 1 / 32])
        self.assertEqual(len(generate(model, document(1))["posters"][0]["panels"]), 1)

    def test_random_seed_and_batching(self):
        model = LayoutModel(config())
        data = document(2)
        other = document(1)["posters"][0]
        other["id"] = "second"
        data["posters"].append(other)
        self.assertEqual(generate(model, data, batch_size=1), generate(model, data, batch_size=2))
        self.assertEqual(generate(model, data, sampling="random", seed=7),
                         generate(model, data, sampling="random", seed=7))

    def test_cli_outside_project(self):
        # Requires installing this package (`pip install -e .`) before running tests.
        cfg = config()
        model = LayoutModel(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(cfg.__dict__), encoding="utf-8")
            (root / "input.json").write_text(json.dumps(document()), encoding="utf-8")
            torch.save({"model.module." + k: v for k, v in model.state_dict().items()}, root / "model.pt")
            subprocess.run([
                sys.executable, "-m", "sciposter_inference", "--config", str(root / "config.json"),
                "--checkpoint", str(root / "model.pt"), "--input", str(root / "input.json"),
                "--output", str(root / "result.json"), "--npy-output", str(root / "result.npy"),
                "--device", "cpu",
            ], cwd=root, check=True, capture_output=True, text=True)
            self.assertEqual(len(json.loads((root / "result.json").read_text())["posters"]), 1)
            import numpy as np
            self.assertEqual(np.load(root / "result.npy", allow_pickle=False).shape, (1, 10, 5))


if __name__ == "__main__":
    unittest.main()
