import json
from pathlib import Path


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
