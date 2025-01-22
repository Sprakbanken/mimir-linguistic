from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import nb_tokenizer
import re
from mimir_linguistic.utils import get_output_dir


def split_sentences(tokens, split_chars=[".", "!", "?"]):
    sentences = []
    sentence = []
    for token in tokens:
        if token in split_chars:
            # add to sentences if not empty
            if len(sentence) > 0:
                sentences.append(sentence)
            sentence = []
        elif re.match("^[^a-zæøåA-ZÆØÅ\d\s:]$", token):
            # skip non-alphanumeric tokens
            continue
        else:
            sentence.append(token)
    return sentences


def average_sentence_length(sentences):
    return sum([len(x) for x in sentences]) / len(sentences)


def percentage_long_words(sentences):
    tokens = sum(sentences, [])
    number_of_tokens = len(tokens)
    if number_of_tokens > 0:
        long_tokens = 0
        for token in tokens:
            if len(token) > 6:
                long_tokens += 1
        return long_tokens * 100 / number_of_tokens
    else:
        return 0


def lix_score(text):
    sentences = split_sentences(text)
    if len(sentences) > 0:
        asl = average_sentence_length(sentences)
        plw = percentage_long_words(sentences)
        return asl + plw
    else:
        return -1


def calculate_lix_scores(df: pd.DataFrame, text_column: str, output_dir: Path):
    df["tokenized_text"] = df[text_column].apply(nb_tokenizer.tokenize)
    df["lix_score"] = df["tokenized_text"].apply(lix_score)
    df[["tokenized_text", "lix_score"]].to_csv(
        output_dir / "scores_per_text.csv", index=False
    )

    results = {}
    tokens_whole_corpus = sum(list(df["tokenized_text"]), [])
    results["lix_score"] = lix_score(tokens_whole_corpus)
    pd.DataFrame(results, index=[0]).to_csv(
        output_dir / "scores_across_texts.csv", index=False
    )


def main():
    """Calculate readability scores on the provided text."""

    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=Path,
        required=True,
        help=".csv file with a column of text to calculate readability scores on",
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

    df = pd.read_csv(args.input_file)

    output_dir = args.output_dir / "readability/"
    output_dir = get_output_dir(output_dir)

    calculate_lix_scores(df=df, text_column=args.text_column, output_dir=output_dir)

    print(f"See results at {output_dir}")
