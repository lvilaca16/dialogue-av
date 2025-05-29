import argparse
from pathlib import Path

from tqdm import tqdm

from src.utils import (move, open_json, verify_complete_sentence,
                       verify_duplicated_entries)


def main(args: argparse.Namespace, verbose: bool = True, dry_run: bool = False):

    iter_bar = tqdm(args.path.glob("*.json"), desc="V/I X/X")

    cnt_valid, cnt_invalid = (0, 0)

    for filepath in iter_bar:

        data = open_json(filepath)[args.type]
        dialogue = [v for _, v in data.items()]

        if len(dialogue) == 0:
            continue

        # Verify each iteration
        iteration_flag = any([verify_complete_sentence(i)[1] for i in dialogue])

        # Verify whole dialogue
        (_, duplicated_flag) = verify_duplicated_entries(dialogue)

        if duplicated_flag or iteration_flag or len(dialogue) < args.min_iterations:

            if args.verbose:
                print(f"Invalid: {filepath.resolve()}")

            if not dry_run:
                # move to subfolder
                move(filepath, filepath.parent.joinpath("invalid"), verbose)

            cnt_invalid += 1

        else:
            if verbose:
                print(f"Valid: {filepath.resolve()}")

            cnt_valid += 1

        iter_bar.set_description(f"V/I ({cnt_valid},{cnt_invalid})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to annotations folder.",
    )

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        default="av_dialogue",
        help="Annotation type (av_dialogue or qa_dialogue).",
    )

    parser.add_argument(
        "--min_iterations",
        type=int,
        help="Minimum number of available iterations.",
        default=10,
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
