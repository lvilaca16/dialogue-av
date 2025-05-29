import argparse
from pathlib import Path
from typing import Union

import clip
import decord
import h5py
import torch
from torchvision.transforms.v2 import Compose, Normalize, ToDtype

from src.beats.BEATs import BEATs, BEATsConfig

RGB_MEAN = (0.48145466, 0.4578275, 0.40821073)
RGB_STD = (0.26862954, 0.26130258, 0.27577711)


def get_audio_encoder(weights_path: Path, device: str = "cpu"):
    # load the pre-trained checkpoints
    ckpt = torch.load(weights_path)
    cfg = BEATsConfig(ckpt["cfg"])

    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])

    return model.to(device).eval()


def sample_audio(audio: decord.AudioReader, n_samples: int, device: str = "cpu"):

    duration = audio._num_samples_per_channel
    space_between_samples = duration // (n_samples + 1)

    if space_between_samples <= 0:
        A = torch.from_numpy(audio[:].asnumpy())

    else:
        sample_ids = []
        for x in range(0, duration, space_between_samples):
            sample_ids.append((x, x + audio.sample_rate))

        sample_ids = sample_ids[1:-1]  # get middle segments

        A = []
        padding_mask = []

        for x, y in sample_ids:
            audio_segment = audio[x:y]

            pad_size = audio.sample_rate - audio_segment.shape[-1]

            if pad_size > 0:
                size = list(audio_segment.shape)
                size[-1] = pad_size

                padding = torch.zeros(tuple(size))
                audio_segment = torch.concatenate((audio_segment, padding), -1)

            mask = torch.zeros(1, audio_segment.shape[-1]).bool()

            if pad_size > 0:
                mask[:, -pad_size:] = True

            A.append(audio_segment)
            padding_mask.append(mask)

        A = torch.concatenate(A, 0)
        padding_mask = torch.concatenate(padding_mask, 0)

    return A.to(device), padding_mask.to(device)


def get_video_encoder(model_name: str = "ViT-L/14", device: str = "cpu"):
    model, _ = clip.load(model_name, device=device)

    transform = Compose(
        [
            ToDtype(torch.float, scale=True),
            Normalize(mean=RGB_MEAN, std=RGB_STD),
        ]
    )

    return model, transform


def sample_video(
    video: decord.VideoReader,
    frame_transform: Union[Compose, None],
    n_samples: int,
    device: str = "cpu",
):
    length = video._num_frame
    space_between_frames = length // (n_samples + 1)

    if space_between_frames == 0:
        frames = video[:].permute(0, 3, 1, 2)

    else:
        frame_ids = [x for x in range(0, length, space_between_frames)][1:-1]
        frames = video.get_batch(frame_ids).permute(0, 3, 1, 2)

    if frame_transform is not None:
        return frame_transform(frames).to(device)

    return frames.to(device)


def main(args: argparse.Namespace, verbose: bool, dry_run: bool):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    decord.bridge.set_bridge("torch")

    audio_encoder = get_audio_encoder(args.beats, device)
    video_encoder, video_transform = get_video_encoder(args.clip, device)

    # Load audio file
    audio = decord.AudioReader(args.path.as_posix(), sample_rate=args.sr)

    video = decord.VideoReader(args.path.as_posix(), height=args.resolution, width=args.resolution)

    A, padding_mask = sample_audio(audio, args.n_samples, device)
    V = sample_video(video, video_transform, args.n_samples, device)

    if not dry_run:
        with torch.no_grad():
            # get audio embeddings
            audio_features = audio_encoder.extract_features(A, padding_mask)[0]

            # get video embeddings
            video_features = video_encoder.encode_image(V)

        label = args.path.stem

        if verbose:
            print(f"Saving embedding: {label}")
            print(f"Audio (shape): {audio_features.shape}")
            print(f"Video (shape): {video_features.shape}")

        with h5py.File(args.output.joinpath("embeddings.hdf5"), "a") as f:
            if "audio" not in f.keys():
                audio_grp = f.create_group("audio")
            else:
                audio_grp = f["audio"]

            if "video" not in f.keys():
                video_grp = f.create_group("video")
            else:
                video_grp = f["video"]

            audio_grp.create_dataset(
                label,
                data=audio_features.detach().cpu(),
                compression="gzip",
                compression_opts=9,
            )

            video_grp.create_dataset(
                label,
                data=video_features.detach().cpu(),
                compression="gzip",
                compression_opts=9,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--beats",
        type=Path,
        required=False,
        default="BEATs_iter3_plus_AS2M.pt",
        help="Path to audio encoder (BEATs) weights.",
    )

    parser.add_argument(
        "--clip",
        type=str,
        required=False,
        default="ViT-L/14",
        help="Name of the CLIP model (ViT-L/14).",
    )

    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to video file (.mp4). Needs to also have an audio track.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        default=Path("."),
        help="Output directory path.",
    )

    parser.add_argument(
        "--sr",
        type=int,
        required=False,
        default=16000,
        help="Sample rate.",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        required=False,
        default=16,
        help="Number of 1 second segments sampled from the audio file.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        required=False,
        default=224,
        help="Resolution for the video frames.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activate verbosity",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Activate verbosity",
    )

    args = parser.parse_args()

    main(args, args.verbose, args.dry_run)
