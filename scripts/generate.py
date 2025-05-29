import argparse
import json
import re

import pandas as pd
import regex
import transformers
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

END_OF_PROMPT = "[/INST]"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def format_prompt(v_cap, a_cap, h_cap, dialogue=True):
    if dialogue:
        return f'<s>[INST] <<SYS>>\n For this video, you are given the following vision captions: "{v_cap.lower()}", audio captions: "{a_cap.lower()}", and the following human verified captions: "{h_cap.lower()}". The video contains crucial information that must be communicated via a dialogue simulating two persons talking about the video contents. Your task is to understand and encode these details into ten dialogue iterations related to the audio, visual, and human-verified captions. The significance between video, audio and human-verified captions is equal. The dialogue should focus on the video content without referencing the term "dialogue". There may be redundancy in vision, audio, and human-verified captions. Please summarise them before encoding into the dialogue. Your attention should be on describing, explaining, or analysing various aspects of the video. Please ensure the dialogue is diverse, high-quality, and reflect the content of the video and audio accurately, offering useful information. The output dialogue should contain only the interactions between actors. In each iteration, the first actor intervention should start with "P1:", while the second should start with "P2:". Generate a JSON object that represents the dialogue. It should contain an array of dialogue iterations. Each iteration should be an object with a property for the text from "P1" and "P2".\n <</SYS>> \n [/INST]'
    else:
        return f'<s>[INST] <<SYS>>\nFor this video, you are given the following vision captions: {v_cap.lower()}, audio captions: {a_cap.lower()}, and the following human verified captions: {h_cap.lower()}. The video contains crucial information that must be communicated via high-quality instructions. Your task is to understand and encode these details into ten pairs of instructions and responses related to the audio, visual, and human-verified captions. The significance of video/audio/human-verified captions is equal. The pairs of questions and answers should focus on the video content without referencing the term "caption". There may be redundancy in vision, audio, and human-verified captions. Please summarise them before encoding into the instruction-response pairs. Your attention should be on describing, explaining, or analysing various aspects of the video, and providing some question-answer pairs. The goal of this exercise is to fine-tune a language model for it to generate accurate and pertinent responses. In each pair, the first line should start with "Q:" and contain an instruction related to the video, while the second line should start with "A:" and provide a response to the instruction. Please ensure your instructions are diverse, high-quality, and reflect the content of the video and audio accurately, offering useful information to the language model.\n <</SYS>> \n [/INST]'


def unpack_llm_output(llm_output, dialogue: bool = True):
    import pdb

    pdb.set_trace()
    if dialogue:
        pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")

        output = pattern.findall(llm_output[0]["generated_text"])[0]

    else:
        for iteration in llm_output:
            data_to_json = {"QA": []}

            _iteration = iteration["generated_text"].replace("\n", "")
            qa_data = re.split("Q: |A: ", _iteration)[1:]  # placeholder

            for i, qa in enumerate(qa_data):
                if i % 2 == 0:
                    data_to_json["QA"].append({"Q": qa})
                else:
                    data_to_json["QA"][-1]["A"] = qa

        output = json.dumps(data_to_json)

    return output


def main(args: argparse.Namespace, dialogue: bool = True):

    data = pd.read_parquet(args.path)

    # Load text generation model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

    for i, row in tqdm(data.iterrows()):

        a_cap = row["audio_captions"]
        v_cap = row["video_captions"]
        h_cap = row["human-verified_captions"]

        prompt = format_prompt(v_cap, a_cap, h_cap, dialogue=dialogue)

        output_text = pipeline(prompt, batch_size=1, num_workers=8)

        # unpack data
        output_text = output_text[0]["generated_text"]
        output_text = output_text[output_text.find(END_OF_PROMPT) + len(END_OF_PROMPT) + 2:]

        if args.verbose:
            print(f"Result: {output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=False,
        default="data/dialogue-av.parquet",
        help="Path to the index file containing ids and captions.",
    )

    parser.add_argument(
        "--dialogue",
        action="store_true",
        help="Flag to output dialogue data, instead of QA.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activate verbosity",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="annotations",
        help="Output directory path.",
    )

    args = parser.parse_args()

    main(args, args.dialogue)
