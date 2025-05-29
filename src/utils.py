import json
import shutil
import string
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.disable_pipe("parser")
nlp.enable_pipe("senter")


INVALID_WORDS = [
    "caption",
    "captions"
]


def open_json(filepath: Path) -> dict:
    """
    Open JSON file

    Arguments:
        filepath -- Path to JSON file

    Raises:
        ValueError: File not found

    Returns:
        Dictionary containing the JSON data
    """

    if filepath.exists():
        with open(filepath.as_posix(), "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"File {filepath.resolve()} not found")


def pairwise(iterable):
    """
    Generate a list of pairwise elements from an iterable.

    Arguments:
        iterable -- Iterable list of elements

    Yields:
        Pairwise elements
    """
    iterator = iter(iterable)
    a = next(iterator, None)

    for b in iterator:
        yield a, b
        a = b


def verify_duplicated_entries(
    dialogue: list, threshold: float = 0.5, min_length: int = 2
) -> Tuple[list, bool]:
    """
    Verify duplicated entries in a list of dialogue iterations.

    Arguments:
        dialogue -- List containing a group of dialogue iterations (string)
        threshold -- duplication threshold
        min_length -- minimum of the unique dialogue (without duplicates)

    Returns:
        Unique entries of the list and duplicated entry flag.
    """
    dialogue = np.array(dialogue)

    (_, mask) = np.unique(dialogue, return_index=True)

    # Unique entries should follow the same order.
    unique_dialogue = dialogue[np.sort(mask)].tolist()

    # percentage of duplicated entries
    percent = 1 - (len(unique_dialogue) / len(dialogue))
    if percent >= threshold or len(unique_dialogue) <= min_length:
        return (unique_dialogue, True)

    return (unique_dialogue, False)


def verify_complete_sentence(
    text, min_length: int = 3
) -> Tuple[Union[None, str], bool]:
    """
    Filters text (input) and finds valid sentences. A complete dialogue turn
    should contains at least one subject, predicate, object or noun, one verb,
    and it should close with punctuation.

    Arguments:
        text -- input text (may contain multiple sentences).

    Keyword Arguments:
        min_length -- minimum character length for the output string (default: {5})

    Returns:
        Filtered list of sentences and a valid dialogue flag.
    """
    doc = nlp(text)
    valid_sents = []

    for sent in doc.sents:
        sent = nlp(sent.text.strip())

        if len(sent.text) == 0:
            return None, True

        # check invalid words
        if string_check(sent.text, INVALID_WORDS):
            return None, True

        # check sentence format and length
        if sent[0].is_title and sent[-1].is_punct:
            valid_tokens = [tok for tok in sent if tok.pos_ != "PUNCT"]

            if len(valid_tokens) >= min_length:  # tokens / sub-sentence
                valid_sents.append(sent.text)

    output_str = " ".join(valid_sents)

    if len(output_str) > min_length:
        return output_str, False
    else:
        return None, True


def filter_floating_punctuation(text: str) -> str:
    """
    Filters floating punctuation

    Arguments:
        text -- input text

    Returns:
        Returns cleaned text
    """
    # remove unwanted data
    text = text.replace("P2", " ")
    text = text.replace("P1", " ")
    text = text.replace("\n", " ")

    text = " " + text + " "  # pad

    for punct in string.punctuation:
        text = text.replace(f" {punct} ", "")

    return text


def string_check(x: str, stop_words: List[str]) -> bool:
    """
    Check the presence of a list of words within a sentence.
    "Search string within another string."

    Arguments:
        x -- Input sentence.
        stop_words -- Words to be searched.

    Returns:
        _description_
    """

    for search_word in stop_words:
        if x.lower().find(search_word.lower()) != -1:
            return True

    return False


def move(src: Path, dst: Path, verbose: bool = False) -> None:
    """
    Move file.

    Arguments:
        src -- source path
        dst -- destination path

    Keyword Arguments:
        verbose -- verbosity flag (default: {False})
    """
    # Create folder is it does not exist
    dst.mkdir(exist_ok=True)

    filename = src.name

    if verbose:
        print(f"SRC: {src.resolve()}")
        print(f"DST {dst.joinpath(filename).resolve()}")

    shutil.move(src.resolve(), dst.joinpath(filename).resolve())
