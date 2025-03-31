from argparse import ArgumentParser
from pathlib import Path
from diversity import compression_ratio
from mimir_linguistic.lexical_diversities.self_bleu import self_bleu_texts
from mimir_linguistic.utils import get_output_dir
import pandas as pd
import nltk

nltk.download("punkt_tab")


def calculate_lexical_diversity_scores(
    df: pd.DataFrame, text_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate self-bleu and compression ratio for each text and all texts together in df.
    Returns two dataframes: one with scores per text and one with scores across all texts.
    Calculates the following scores:
        - self-bleu
        - compression ratio
    """
    df = df.copy()
    df["text_str_len"] = df[text_column].apply(len)
    df["self_bleu"] = self_bleu_texts(df[text_column])
    df["compression_ratio"] = df[text_column].apply(
        lambda text: compression_ratio([text], "gzip")
    )

    scores_per_text = df[
        [
            "text_str_len",
            "compression_ratio",
            "self_bleu",
        ]
    ]

    results = {}
    texts_concat = " ".join(df[text_column])
    results["str_len"] = len(texts_concat)
    results["compression_ratio"] = compression_ratio(df[text_column], "gzip")
    results["compression_ratio_mean"] = scores_per_text["compression_ratio"].mean()
    results["self_bleu"] = scores_per_text["self_bleu"].mean()

    scores_across_texts = pd.DataFrame(results, index=[0])
    return scores_per_text, scores_across_texts


def main():
    """Calculate lexical diversity scores on the provided text."""

    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=Path,
        required=True,
        help=".csv file with a column of text to calculate diversity scores on",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Name of text column in input csv",
    )

    parser.add_argument(
        "--output_dir", "-o", type=Path, required=True, help="directory to save outputs"
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Input file {args.input_file} does not exist")
        exit()

    output_dir = args.output_dir / "lexical_diversity/"
    output_dir = get_output_dir(output_dir)

    df = pd.read_csv(args.input_file)
    scores_per_text, scores_across_texts = calculate_lexical_diversity_scores(
        df=df, text_column=args.text_column
    )

    scores_per_text.to_csv(output_dir / "scores_per_text.csv", index=False)

    scores_across_texts.to_csv(output_dir / "scores_across_texts.csv", index=False)
    print(f"See results at {output_dir}")
