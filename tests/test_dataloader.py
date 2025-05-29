import unittest
from pathlib import Path

from torch.utils.data import DataLoader
import torch
from src.dataloader import DialogueAV


METADATA_PATH = Path("data/test/test.tar.gz")
EMBEDDINGS_PATH = Path("data/test/test.hdf5")
DATA_PATH = Path("data/media")


class TestDataloader(unittest.TestCase):

    def test_dataloader_with_media(self):
        """
        Evaluate Dialogue-AV Dataset: check expected shapes
        """
        dataset = DialogueAV(METADATA_PATH, DATA_PATH, "qa_dialogue")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        (video, audio), (x_text, _), (y_text, _) = next(iter(dataloader))

        self.assertEqual(video.shape, torch.Size((1, 16, 3, 224, 224)))
        self.assertEqual(audio.shape, torch.Size((1, 16, 64, 100)))
        self.assertEqual(x_text.shape[-1], 128)
        self.assertEqual(y_text.shape[-1], 128)

    def test_dataloader_with_embeddings(self):
        """
        Evaluate Dialogue-AV Dataset (using embeddings): check expected shapes
        """
        dataset = DialogueAV(
            METADATA_PATH,
            DATA_PATH,
            "qa_dialogue",
            embeddings_path=EMBEDDINGS_PATH,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        (video, audio), (x_text, _), (y_text, _) = next(iter(dataloader))

        self.assertEqual(video.shape, torch.Size((1, 16, 768)))
        self.assertEqual(audio.shape, torch.Size((1, 16, 48, 768)))
        self.assertEqual(x_text.shape[-1], 128)
        self.assertEqual(y_text.shape[-1], 128)


if __name__ == "__main__":
    unittest.main()
