from argparse import ArgumentParser
from pathlib import Path
from diversity import compression_ratio, ngram_diversity_score
from lexical_diversity import lex_div as ld
from mimir_linguistic.lexical_diversities.self_bleu import self_bleu_texts
from mimir_linguistic.lexical_diversities.stopwords import stop_words
from mimir_linguistic.utils import get_output_dir
import pandas as pd
import spacy
import nltk

nltk.download("punkt_tab")


def stopword_density(tokenized_text: list[str]) -> float:
    return len([t for t in tokenized_text if str(t).lower() in stop_words]) / len(
        tokenized_text
    )


def calculate_lexical_diversity_scores(
    df: pd.DataFrame, text_column: str, ns: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate lexical diversity scores for each text and all texts together in df.
    Returns two dataframes: one with scores per text and one with scores across all texts.
    Calculates the following scores:
        - self-bleu
        - compression ratio
        - part-of-speech compression_ratio
        - n_gram_diversity_score
        - type_token_ratio
        - moving_average_type_token_ratio
        - stopword_density
    """
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

    df["self_bleu"] = self_bleu_texts(df[text_column])

    results["simple_ttr"] = ld.ttr(texts_concat)
    results["moving_average_ttr"] = ld.mattr(texts_concat)
    results["number_of_tokens"] = len(ld.tokenize(texts_concat))
    results["str_len"] = len(texts_concat)

    df["stopword_density"] = df.tokenized_text.apply(stopword_density)
    results["stopword_density"] = stopword_density(
        [token for text in df.tokenized_text for token in text]
    )
    scores_per_text = df[
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
    ]
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
    scores_per_text, scores_across_texts = calculate_lexical_diversity_scores(
        df=df, text_column=args.text_column, ns=args.ns
    )

    scores_per_text.to_csv(output_dir / "scores_per_text.csv", index=False)

    scores_across_texts.to_csv(output_dir / "scores_across_texts.csv", index=False)
    print(f"See results at {output_dir}")
