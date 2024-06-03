from argparse import ArgumentParser
from datasets import IterableDataset, load_dataset
from transformers import set_seed
from collections import Counter, defaultdict
import json
import pandas as pd
from pathlib import Path
import random
import re
from utils import get_output_dir, arglist_to_kwarg_dict


def tokenize(text: str) -> list[str]:
    return re.split(r"\W+", text)


def sample_x_random_n_grams(
    dataset: IterableDataset, x: int, n: int, save_doc_info: bool
) -> dict[str, list]:
    start_prompts = defaultdict(list)
    for e in dataset:
        if len(start_prompts["prompt"]) == x:
            break
        texts = e["text"].split("\n")
        first_n_tokens = [
            [token for token in tokenize(text) if token][:n] for text in texts
        ]
        first_n_tokens = [
            " ".join(token_list)
            for token_list in first_n_tokens
            if len(token_list) == n
        ]
        if first_n_tokens:
            text = random.choice(first_n_tokens)
            start_prompts["prompt"].append(text)
            if save_doc_info:
                start_prompts["doc_id"].append(e["id"])
                start_prompts["doc_type"].append(e["doc_type"])
        else:
            continue

    return start_prompts


def sample_x_most_frequent_n_grams(
    dataset: IterableDataset, x: int, n: int, save_doc_info: bool
) -> dict[str, list]:
    counter = Counter()
    if save_doc_info:
        ngrams_to_docs = defaultdict(set)

    for e in dataset:
        text = e["text"]
        texts = text.split("\n")
        first_n_tokens = [
            [token for token in tokenize(text) if token][:n] for text in texts
        ]
        first_n_tokens = [
            " ".join(token_list)
            for token_list in first_n_tokens
            if len(token_list) == n
        ]
        counter.update(first_n_tokens)

        if save_doc_info:
            for n_gram in first_n_tokens:
                ngrams_to_docs[n_gram].add((e["id"], e["doc_type"]))

    if len(counter) == 0:
        n_grams, counts = [], []
    n_grams, counts = zip(*counter.most_common(x))

    if save_doc_info:
        ids_types = [ngrams_to_docs[e] for e in n_grams]
        doc_ids, doc_types = zip(*[[e for e in zip(*list(set_))] for set_ in ids_types])

        return {
            "prompt": n_grams,
            "count": counts,
            "doc_ids": doc_ids,
            "doc_types": doc_types,
        }

    return {"prompt": n_grams, "count": counts}


def main():
    """Sample starts of sentences from dataset"""
    parser = ArgumentParser()
    parser.add_argument(
        "--source_dataset",
        required=True,
        help="Huggingface hub dataset id or path to local datasets dataset",
    )
    parser.add_argument("--split", help="Dataset split to use", default="validation")
    parser.add_argument(
        "--x",
        "-x",
        type=int,
        required=True,
        help="Number of sentence starters to pick out",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        required=True,
        help="Number of words in each sentence starter",
    )
    parser.add_argument(
        "--sampling_strategy", "-ss", choices=["frequent", "random"], default="random"
    )
    parser.add_argument("--seed", type=int, help="Seed for random sampling", default=42)
    parser.add_argument(
        "--filters",
        nargs="+",
        default=[],
        help="Sequence of feature=value pairs to filter dataset rows by",
    )
    parser.add_argument(
        "--save_doc_info",
        action="store_true",
        default=False,
        help="Save doc_id and doc_type for all selected text starters if flagged",
    )
    parser.add_argument(
        "--output_dir", "-o", type=Path, required=True, help="directory to save outputs"
    )
    args = parser.parse_args()
    set_seed(args.seed)

    dataset = load_dataset(
        args.source_dataset, split=args.split, streaming=True, trust_remote_code=True
    )

    filters = arglist_to_kwarg_dict(args.filters)
    if filters:
        filter_string = "lambda x : " + " and ".join(
            f"x['{k}'] == '{v}'" for k, v in filters.items()
        )
        dataset = dataset.filter(eval(filter_string))

    output_dir = args.output_dir / "text_starters/"
    output_dir = get_output_dir(output_dir)

    if args.sampling_strategy == "random":
        prompts_dict = sample_x_random_n_grams(
            dataset=dataset, x=args.x, n=args.n, save_doc_info=args.save_doc_info
        )
    else:
        prompts_dict = sample_x_most_frequent_n_grams(
            dataset=dataset, x=args.x, n=args.n, save_doc_info=args.save_doc_info
        )
    if len(prompts_dict["prompt"]) < args.x:
        print(
            f"Could not find {args.x} starter {args.n}-grams, only {len(prompts_dict)}"
        )

    with open(output_dir / "text_starters.json", "w+") as f:
        json.dump(prompts_dict, f, indent=4, default=list, ensure_ascii=False)
    with open(output_dir / "args.json", "w+") as f:
        json.dump(vars(args), f, default=str, indent=4, sort_keys=True)

    pd.DataFrame(prompts_dict).to_csv(output_dir / "text_starters.csv", index=False)
