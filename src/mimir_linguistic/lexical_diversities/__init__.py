from argparse import ArgumentParser
from pathlib import Path
from diversity import compression_ratio, ngram_diversity_score
from lexical_diversity import lex_div as ld
from mimir_linguistic.lexical_diversities.SelfBleu import SelfBleu
from mimir_linguistic.utils import get_output_dir
import pandas as pd
import spacy


def calculate_lexical_diversity_scores(
    df: pd.DataFrame, text_column: str, ns: list[int], output_dir: Path
):
    nlp = spacy.load("nb_core_news_lg")
    docs = pd.Series(nlp.pipe(df[text_column]))

    df["tokenized_text"] = docs.apply(list)
    df["parts_of_speech"] = docs.apply(lambda x: " ".join([e.tag_ for e in x]))

    results = {}
    results["compression_ratio"] = compression_ratio(df[text_column], "gzip")
    results["pos_compression_ratio"] = compression_ratio(df.parts_of_speech, "gzip")

    for n in ns:
        results[f"{n}_gram_diversity_score"] = ngram_diversity_score(df[text_column], n)

    df["simple_ttr"] = df[text_column].apply(ld.ttr)
    df["moving_average_ttr"] = df[text_column].apply(ld.mattr)
    df["number_of_tokens"] = df.tokenized_text.apply(len)
    df["text_str_len"] = df[text_column].apply(len)

    texts_concat = " ".join(df[text_column])

    # Self-Bleu (is_fast=False does not rely on sampling)
    df["text_id"] = range(len(df))
    selfbl = SelfBleu(test_text=list(df[text_column]))
    df["self_bleu"] = df["text_id"].apply(selfbl.get_bleu_one_hypothesis)
    results["self_bleu"] = selfbl.get_score(is_fast=False)

    results["simple_ttr"] = ld.ttr(texts_concat)
    results["moving_average_ttr"] = ld.mattr(texts_concat)
    results["number_of_tokens"] = len(ld.tokenize(texts_concat))
    results["str_len"] = len(texts_concat)

    with open("src/mimir_linguistic/lexical_diversities/stopwords.txt") as f:
        stopwords = set(f.read().split("\n"))

    def stopword_density(tokenized_text):
        return len([t for t in tokenized_text if str(t).lower() in stopwords]) / len(
            tokenized_text
        )

    df["stopword_density"] = df.tokenized_text.apply(stopword_density)
    results["stopword_density"] = stopword_density(
        [token for text in df.tokenized_text for token in text]
    )

    df[
        [
            "tokenized_text",
            "parts_of_speech",
            "simple_ttr",
            "moving_average_ttr",
            "number_of_tokens",
            "text_str_len",
            "stopword_density",
            "self_bleu",
        ]
    ].to_csv(output_dir / "scores_per_text.csv", index=False)

    df = pd.DataFrame(results, index=[0])
    df.to_csv(output_dir / "scores_across_texts.csv", index=False)


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
        "--ns",
        type=int,
        nargs="+",
        default=[3, 4],
        help="Ns for N-gram diversity score",
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
    calculate_lexical_diversity_scores(
        df=df, text_column=args.text_column, ns=args.ns, output_dir=output_dir
    )

    print(f"See results at {output_dir}")
