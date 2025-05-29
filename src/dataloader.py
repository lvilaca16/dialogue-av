from pathlib import Path
import tarfile
from typing import Literal, Tuple, Union
import json
import h5py

import decord
import numpy as np
import torch
from torchvision.transforms.transforms import Compose, Lambda
from transformers import GPT2Tokenizer
from torchaudio.transforms import MelSpectrogram


from .utils import pairwise


decord.bridge.set_bridge("torch")


class DialogueAV(torch.utils.data.Dataset):
    """Load sentence pair (sequential or random order) from corpus"""

    def __init__(
        self,
        metadata_path: Path,
        data_path: Path,
        task: Literal["qa_dialogue", "av_dialogue"] = "av_dialogue",
        tokenizer: Union[GPT2Tokenizer, None] = None,
        max_length: int = 128,
        embeddings_path: Union[Path, None] = None,
        n_samples: int = 16,
        frame_transform: Union[Compose, None] = Lambda(lambda x: x / 255.0),
        resolution: int = 224,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        win_length: float = 0.025,
        hop_length: float = 0.01,
        n_mels: int = 64,
        audio_sample_size: int = 1,
    ):
        assert task in ["qa_dialogue", "av_dialogue"], f"Invalid task ({task})"

        assert (metadata_path.exists()), f"Invalid metadata ({metadata_path})"
        assert data_path.exists(), f"Invalid data path ({data_path})"

        self.tar_fp = tarfile.open(metadata_path.as_posix(), "r")

        # Validate existent samples
        self.samples = []
        samples = self.tar_fp.getmembers()

        for i, sample in enumerate(samples):
            annotation = self.extract_and_read(sample)

            if len(annotation[task]) > 0:
                self.samples.append(sample)

        self.task = task

        self.embeddings_path = embeddings_path

        if embeddings_path is not None:
            assert embeddings_path.exists(), "Invalid path to embeddings"

            self.embeddings = h5py.File(embeddings_path, "r")

            assert "audio" in self.embeddings.keys(), "Missing audio group in hdf5"
            assert "video" in self.embeddings.keys(), "Missing video group in hdf5"

        # data folders
        self.audio_dir = data_path.joinpath("audio")
        self.video_dir = data_path.joinpath("video")

        self.n_samples = n_samples

        # video properties
        self.resolution = resolution
        self.frame_transform = frame_transform

        # audio properties
        self.sr = sample_rate

        self.audio_sample_size = int(audio_sample_size)
        self.spectrogram_shape = (n_mels, int(audio_sample_size / hop_length))

        self.audio_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(win_length * sample_rate),
            hop_length=int(hop_length * sample_rate),
            center=True,
            pad_mode="constant",
            power=2.0,
            norm="slaney",
            n_mels=n_mels
        )

        # # text properties
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        tokenizer.padding_side = "right"
        tokenizer.model_max_length = max_length
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.tokenizer = tokenizer

        super().__init__()

    def __len__(self) -> int:
        return len(self.samples)

    def extract_and_read(self, x):
        return json.loads(self.tar_fp.extractfile(x).read())

    def get_idx(self, idx: int) -> str:
        """
        Extract sample from tar.gz file
        """
        return self.extract_and_read(self.samples[idx])

    def __getitem__(self, idx: int):
        # unpack
        metadata = self.get_idx(idx)
        annotation = metadata[self.task]

        # I x D
        text_x, text_y = self._get_text(annotation)

        if self.embeddings_path is None:

            audio_path = self.audio_dir.joinpath(f"{metadata['filename']}.wav")
            video_path = self.video_dir.joinpath(f"{metadata['filename']}.mp4")

            if not audio_path.exists() or not video_path.exists():
                raise FileNotFoundError("File not found.")

            # T x H x W x C
            video, timestamps = self.get_video(video_path)

            # T x B x D
            audio = self.get_audio(audio_path, timestamps)

        else:
            audio = self.embeddings["audio"].get(metadata["filename"])[:]
            video = self.embeddings["video"].get(metadata["filename"])[:]

        return (
            (video, audio),
            (text_x["input_ids"], text_x["attention_mask"]),
            (text_y["input_ids"], text_y["attention_mask"]),
        )

    def get_video(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:

        video = decord.VideoReader(
            path.as_posix(), width=self.resolution, height=self.resolution
        )

        length = video._num_frame
        frame_interval = length // (self.n_samples + 1)

        if frame_interval == 0:
            frame_ids = list(np.arange(0, length))
            frames = video[:].permute(0, 3, 1, 2)

        else:
            frame_ids = [x for x in range(0, length, frame_interval)][1:-1]
            frames = video.get_batch(frame_ids).permute(0, 3, 1, 2)

        timestamps = video.get_frame_timestamp(frame_ids)
        timestamps = np.round(timestamps, 2)

        if self.frame_transform is not None:
            frames = self.frame_transform(frames)

        return frames, timestamps

    def get_audio(self, path: Path, timestamps: np.ndarray) -> np.ndarray:
        audio = decord.AudioReader(path.as_posix(), sample_rate=self.sr)

        # apply function element wise
        sample_ids = np.vectorize(audio._time_to_sample)(timestamps)

        # each frame receives one second of audio
        sample_ids[:, 1] = sample_ids[:, 0] + (self.sr * 1)

        mel_spectrograms = []
        for x, y in list(sample_ids):
            A = audio[x:y].squeeze()

            pad_size = (self.sr * self.audio_sample_size) - A.shape[0]
            A = self.add_padding(A, pad_size, 0)

            S = self.audio_transform(A)

            # limit to (n_mels, (window_for_each_frame / hop_length))
            S = S[: self.spectrogram_shape[0], : self.spectrogram_shape[1]]
            mel_spectrograms.append(S)

        return torch.stack(mel_spectrograms)

    def _get_text(self, input: dict) -> Tuple:
        text = list(input.values())

        # validation (ensure we have pairs)
        if len(text) % 2 != 0:
            text = text[:-1]

        if self.task == "qa":
            # Assumes pairs of Q and A (no past QAs are used)
            x = [j for i, j in enumerate(text) if i % 2 == 0]
            y = [j for i, j in enumerate(text) if i % 2 == 1]
            assert len(x) == len(y)

        else:
            # Assumes pairs of dialogue turns as y_true (no past turns are used)
            x = [i[0] for i in pairwise(text)]
            y = [i[1] for i in pairwise(text)]
            assert len(text) - 1 == len(y)

        x = self.tokenizer(x, return_tensors="pt", padding="max_length")
        y = self.tokenizer(y, return_tensors="pt", padding="max_length")

        return x, y

    def add_padding(self, mat: torch.tensor, pad_size: int, dim: int = 0) -> np.array:
        size = list(mat.shape)
        size[dim] = pad_size

        pad_values = torch.zeros(tuple(size))
        return torch.concatenate((mat, pad_values), dim)
