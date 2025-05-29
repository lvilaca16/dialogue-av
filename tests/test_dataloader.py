import unittest
from pathlib import Path

from torch.utils.data import DataLoader
import torch
from src.dataloader import DialogueAV


INDEX_PATH = Path("data/test/test.parquet")
DATA_PATH = Path("data/media")


class TestDataloader(unittest.TestCase):

    def test_dataloader(self):
        """
        Evaluate Dialogue-AV Dataset: check expected shapes
        """
        dataset = DialogueAV(INDEX_PATH, DATA_PATH, "question_answering")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        video, audio, (x_text, y_text) = next(iter(dataloader))

        self.assertEqual(video.shape, torch.Size((1, 8, 224, 224, 3)))
        self.assertEqual(audio.shape, torch.Size((1, 8, 64, 100)))
        self.assertEqual(x_text.shape[-1], 512)
        self.assertEqual(y_text.shape[-1], 512)


if __name__ == "__main__":
    unittest.main()
