# DialogueAV: a Dialogue-attended Audiovisual Dataset

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This is the official release of the code for **DialogueAV: a Dialogue-attended Audiovisual Dataset**. Dialogue-AV is a benchmarking dataset with ~280k video clips. Each clip has two dialogue-based descriptions: a Question-Answering Dialogue (QDA) with ten question-answer pairs and a simulated conversation between two "humans" discussing the video.

The dialogues come from human-created captions in SOTA benchmarking datasets and machine-generated captions. We use verified annotations from these top datasets, focusing solely on describing the audiovisual content.

## Methodology

In the Dialogue-AV sample we present next, the input consists of a video containing an audio track along with its original text captions (1). The output is a series of dialogue turns that describe the video's content. We process the input video using audio and video captioners (2), which generate text descriptions corresponding to each modality. All captions, including the original, are transformed into dialogue (4) and question-answer (5) conversations that articulate the audiovisual content.

![](docs/figures/example_dialogue.png)

Annotations in (4) and (5) undergo automatic validation (3) before they are accepted into Dialogue-AV. In the automatic validation step (3), accepted samples must:

1. Include between 5 and 20 dialogue turns;
2. Each dialogue turn must have at least one complete sentence. A complete sentence requires at least 1 subject, predicate, object or noun, and 1 verb; it should end with appropriate punctuation and begin with a named character. Additionally, each complete sentence must contain a minimum of 3 words after removing punctuation (avoid simple sentences as "It rains.").
3. Avoid using the terms "caption(s)" or "dialogue(s)", thereby eliminating references to the original prompt (previously fed to Llama2 in Step 3 - previous figure).

To instruct the LLM into generating our dialogue-based data (3) we developed the following prompts:

### Dialogue Prompt

> For this video, you are given the following vision captions: "{}", audio captions: "{}", and the following human-verified captions: "{}". The video contains crucial information that must be communicated via a dialogue simulating two persons talking about the video contents. Your task is to understand and encode these details into ten dialogue iterations related to the audio, visual, and human-verified captions. The significance between video, audio and human-verified captions is equal. The dialogue should focus on the video content without referencing the terms "dialogue" and "caption". There may be redundancy in vision, audio, and human-verified captions. Please summarise them before encoding into the dialogue. Your attention should be on describing, explaining, or analysing various aspects of the video. Please ensure the dialogue is diverse, high-quality, and reflect the content of the video and audio accurately, offering useful information. The output dialogue should contain only the interactions between actors. In each iteration, the first actor intervention should start with "P1:", while the second should start with "P2:". Generate a JSON object that represents the dialogue. It should contain an array of dialogue iterations. Each iteration should be an object with a property for the text from "P1" and "P2".

### QA-based Dialogue Prompt

> For this video, you are given the following vision captions: "{}", audio captions: "{}", and the following human-verified captions: "{}". The video contains crucial information that must be communicated via high-quality instructions. Your task is to understand and encode these details into ten pairs of instructions and responses related to the audio, visual, and human-verified captions. The significance of video, audio and human-verified captions is equal. The pairs of questions and answers should focus on the video content without referencing the terms "dialogue" and "caption". There may be redundancy in vision, audio, and human-verified captions. Please summarise them before encoding into the instruction-response pairs. Your attention should be on describing, explaining, or analysing various aspects of the video, and providing some question-answer pairs. The goal of this exercise is to fine-tune a language model for it to generate accurate and pertinent responses. In each pair, the first line should start with "Q:" and contain an instruction related to the video, while the second line should start with "A:" and provide a response to the instruction. Please ensure your instructions are diverse, high-quality, and reflect the content of the video and audio accurately, offering useful information to the language model.

## Installation

The minimum framework requirement for this codebase is `Python 3.9`. To leverage CUDA, we used the following framework: `CUDA 12.1`, `NVIDIA Driver v 560.35.03` and `CuDNN 8.9.2`.

We provide an Anaconda environment files (`environment.yml`) to streamline setting up the working environment.

```bash
conda env create -f environment.yml
```

Afterwards, install Spacy's english model:
```bash
python -m spacy download en_core_web_sm
```

## Data

We utilized Dialogue-AV, which provides two versions of dialogues discussing the audiovisual content.

1. **Question answering (QA)**: each pair of QA dialogue turns is composed of a question and its answer about the audio and video contents.

2. **Dialogue**: Dialogue simulating a conversation between two people discussing audiovisual contents.

These dialogue annotations and the audiovisual embeddings for the test set can be found in our **[Zenodo repository](https://zenodo.org/uploads/13897941)**.

> **Download the *.tar.gz files** for all splits (train, validation and test) and place them inside the `data/<split>` directories.

Each *.tar.gz file contains the annotations in the form of .JSON files, where every video segment has a .JSON annotation.

## Usage

### Dataloading

To use our dataset in your experiments, we provide a Torch dataloader to load video data its respective annotations.

For each dialogue sample we load pairs of iterations. The goal is to obtain a model that predicts the next dialogue turn (y) based on the previous one (x). In the QA version, we use non-overlapping dialogue pairs, while in the dialogue version we use overlapping pairs.

We supply the Torch dataloader to ease the process of setting up the dataset:

```python
from torch.data.utils import Dataloader

from src.dataloader import DialogueAV

METADATA_PATH="<path to *.tar.gz file with the annotations>"
DATA_PATH="<path to folder where you have the .mp4 (video) and .wav (audio) files>"
EMBEDDINGS_PATH="<path to the *.hdf5 file with the audiovisual embeddings (optional)>"

dataset = DialogueAV(METADATA_PATH, DATA_PATH, "<flag to select the DialogueAV version>", embeddings_path=EMBEDDINGS_PATH)

dataset = DialogueAV(INDEX_PATH, DATA_PATH, )
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for (video, audio), (x_text_ids, x_text_att_mask), (y_text_ids, _) in dataloader:
  ...

```

### Replicating the dataset construction pipeline

If you would like to **replicate our pipeline** for building Dialogue-AV, you'll need to do the following steps:

1. Download the original video files (from Youtube or directly downloading the from the official sources) and preprocess each video file using the following rules:
   - Download all videos in the maximum possible resolution up to 720p.
   - Audio channel: 1 channel, 192kbps, PCM with 16-bit (audio codec), 16kHz.
   - Video channel: H.264, 25 FPS.


Afterwards you'll need to also clone the captioning submodule. In the captioning module you'll find all the pertinent information to deploy the audio and video captioning models we used.

```bash
# Add this flag to the clone cmd to also clone the captioning submodule
git clone --recurse-submodules git@github.com:lvilaca16/dialogue-av.git
```

2. Follow the instructions on `captioners/README.md` to understand how to use the caption models.

3. Generate dialogue. You'll need the audiovisual and the human-validated captions.

These captions are provided in the `data/dialogue-av.parquet` file. The `dialogue-av.parquet` file provides references to the original audiovisual contents, including their original datasets and Youtube URLs (when available).

  ```python
  # To generate the dialogue version
  python scripts/generate.py --path=data/dialogue-av.parquet --dialogue

  # To generate the QA dialogue version
  python scripts/generate.py --config_path=data/dialogue-av.parquet
  ```

Parse the previous output and create a .JSON file for each sample with the following structure:
```json
{
  "filename": "audiocaps_096oTVzc5Gs_30.000_40.000",
  "url": "https://www.youtube.com/watch?v=096oTVzc5Gs",
  "split": "test",
  "video_id": "096oTVzc5Gs",
  "segment_id": "106977",
  "start": "30.000",
  "end": "40.000",
  "dataset": "audiocaps",
  "captions": {
    "human-captions": "...",
    "audio_captions": "...",
    "video_captions": "..."
  }
  "av_dialogue": {
    "0": "...",
    "1": "...",
    ...
  }
  "qa_dialogue": {
    "0": "...",
    "1": "...",
    ...
  }
}
```

4. Clean and validate machine-generated data. This script assumes that you have your data in a single directory where all .JSON files contain the dialogue annotations from the previous step.

  ```python
  # To clean
  python scripts/validate.py \
    --path <path_to_annotations_folder> \
    --min_iterations 5 \
    --verbose
  ```

### Generating Embeddings (optional)

Instead of using latent representations, the main goal of our dataset is to leverage the audiovisual contents directly. While we do not provide raw video data, we do offer audio and video embeddings derived from ViT-L/14 (CLIP) and BEATS$_{iter3+}$.

For transparency, we provide the Python script that was used to generate audiovisual embeddings.

Before running the script you need to download the weights for the audio encoder (BEATS) from its [official repository](https://github.com/microsoft/unilm/tree/master/beats) and place it in the root of the repository. We used the iteration $iter3+$ of the pre-trained model.

You can run the script as follows:

  ```python
  python scripts/get_embeddings.py \
    --beats=BEATs_iter3_plus_AS2M.pt \
    --clip=ViT-L/14 \
    --path=<path_to_mp4_video_file> \
    --verbose
  ```

The script will create an .hdf5 file that contains the audio and video embeddings. Afterwards, the embeddings are easily indexed using the filename.

```python
with h5py.File(<path_to_embeddings>, "r") as f:
    audio = f["audio"].get(<filename>)[:]
    video = f["video"].get(<filename>)[:]
```


## Citations

```
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```

## Correspondence and Maintenance

Any feedback is appreciated. If you observed any issues, please contact us. All project-related issues and feature requests should be submitted through our GitHub Issues page.
