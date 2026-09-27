import copy
import unittest

from sciposter_inference.schema import ModelConfig, validate_input


def config(**kwargs):
    return ModelConfig(d_model=8, nhead=2, dim_feedforward=16, num_layers=1, **kwargs)


def document(count=2):
    return {"posters": [{"id": "example", "panels": [
        {"category": i % 5, "text_len": 2, "text_ratio": 3,
         "figure_count": 1, "figure_ratio": 4} for i in range(count)
    ]}]}


class SchemaTests(unittest.TestCase):
    def test_names_and_padding(self):
        data = document()
        data["posters"][0]["panels"][0]["category"] = "Title"
        panels = validate_input(data, config())[0]["panels"]
        self.assertEqual(panels[0]["category"], 0)

    def test_bad_features_rejected(self):
        for field, value in [("category", 5), ("text_len", 0.5), ("text_ratio", True),
                             ("figure_count", 11)]:
            with self.subTest(field=field, value=str(value)[:20]):
                data = document()
                data["posters"][0]["panels"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_input(data, config())

    def test_panel_limits_and_duplicate_ids(self):
        for count in (0, 11):
            with self.assertRaises(ValueError):
                validate_input(document(count), config())
        data = document()
        data["posters"].append(copy.deepcopy(data["posters"][0]))
        with self.assertRaises(ValueError):
            validate_input(data, config())

    def test_training_config_conversion(self):
        raw = {
            "backbone": {"encoder_layer": {
                "_target_": "trainer.models.transformer_utils.Block", "d_model": 256,
                "nhead": 8, "dim_feedforward": 2048, "norm_first": True, "batch_first": True,
            }, "num_layers": 4},
            "model": {"_target_": "trainer.models.bart.BART", "_partial_": True},
            "dataset": {"_target_": "trainer.datasets.posterGlobal.posterGlobalDataset", "max_seq_length": 10},
            "data": {"num_bin_bboxes": 32, "bbox_quantization": "linear", "shared_bbox_vocab": "xywh",
                     "var_order": "c-w-h-x-y", "pad_until_max": True,
                     "special_tokens": ["pad", "bos", "eos", "mask"]},
            "extra_model_params": {"multi_decoder": False, "category_loss": "???"},
        }
        parsed = ModelConfig.from_mapping(raw)
        self.assertEqual((parsed.d_model, parsed.dim_feedforward), (168, 1344))
        self.assertFalse(parsed.multi_decoder)
        raw["extra_model_params"]["use_size_position_decoders"] = True
        with self.assertRaises(ValueError):
            ModelConfig.from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
