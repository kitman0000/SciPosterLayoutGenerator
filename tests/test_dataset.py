import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sciposter_inference.dataset import load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "posterGlobal-max10/raw"
        (self.raw / "poster_meta").mkdir(parents=True)
        (self.raw / "poster_id_list.txt").write_text("train\nfirst\nsecond\n")
        (self.raw / "test_poster_id.txt").write_text("second\nfirst\n")
        (self.raw / "train_poster_id.txt").write_text("train\n")
        boxes = np.zeros((3, 10, 5))
        boxes[:, :, 0] = 5
        boxes[:, :2, 0] = [0, 2]
        np.save(self.raw / "poster_panel_position.npy", boxes)
        for name, length, figures in (("train", 100, 10), ("first", 25, 2), ("second", 50, 5)):
            # Intentionally reverse section keys: insertion order must be retained.
            rows = {"9": {"textlen": length, "figures": figures, "textRatio": 0.25, "figRatio": 0.35},
                    "1": {"textlen": 0, "figures": 0, "textRatio": 0.0, "figRatio": 0.0}}
            (self.raw / "poster_meta" / f"{name}.json").write_text(json.dumps({"section": rows}))

    def test_order_and_global_normalization(self):
        data = load_dataset(self.root)
        self.assertEqual([p["id"] for p in data["posters"]], ["first", "second"])
        self.assertEqual(data["posters"][0]["panels"][0], {
            "category": 0, "text_len": 2, "figure_count": 2,
            "text_ratio": 2, "figure_ratio": 3,
        })

    def test_single_selection_does_not_change_features(self):
        all_posters = load_dataset(self.root)["posters"]
        one = load_dataset(self.raw, poster_ids=["second"])["posters"]
        self.assertEqual(one, all_posters[1:])
        self.assertEqual(load_dataset(self.raw.parent, limit=1)["posters"], all_posters[:1])
        with self.assertRaises(ValueError):
            load_dataset(self.root, poster_ids=["train"])

    def test_short_metadata_zero_padded(self):
        path = self.raw / "poster_meta/first.json"
        doc = json.loads(path.read_text())
        del doc["section"]["1"]
        path.write_text(json.dumps(doc))
        panel = load_dataset(self.root)["posters"][0]["panels"][1]
        self.assertEqual(panel, {"category": 2, "text_len": 0, "text_ratio": 0, "figure_count": 0, "figure_ratio": 0})


if __name__ == "__main__":
    unittest.main()
