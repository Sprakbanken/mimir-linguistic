from argparse import ArgumentParser
from collections import defaultdict
from itertools import cycle
import json
import pandas as pd
from pathlib import Path
import random
from string import punctuation
from transformers import GenerationConfig
from utils import (
    get_model_and_tokenizer,
    get_quantized_model_and_tokenizer,
    arglist_to_kwarg_dict,
    batch_generate,
    get_output_dir,
    get_first_predicted_word_only,
)


def load_dataset(only_grammar: bool, num_examples: int) -> dict[str, list[str]]:
    with open("src/analogies/norwegian-analogies/norwegian-analogies.txt") as f:
        lines = [line.strip() for line in f.readlines()]
    dataset = defaultdict(list)
    for line in lines:
        if line.startswith(":"):
            group_id = line[2:]
            continue
        if only_grammar and not group_id.startswith("gram"):
            continue
        dataset[group_id].append(line)
    if num_examples:
        examples_per_key = num_examples // len(dataset.keys())
        rest = num_examples - (len(dataset.keys()) * examples_per_key)
        for i, key in enumerate(dataset):
            if not i:
                dataset[key] = dataset[key][: examples_per_key + rest]
            else:
                dataset[key] = dataset[key][:examples_per_key]
    return dataset


def get_n_shots(dataset: dict[str, list[str]], n: int, seed) -> list[str]:
    """Remove n examples from dataset and return"""
    random.seed(seed)
    examples = []
    cyc = cycle(dataset.keys())
    while len(examples) < n:
        k = next(cyc)
        examples.append(dataset[k].pop(random.randint(0, len(dataset[k]))))
    return examples


def format_shots(shots: list[str], delimiter: str) -> str:
    lines = []
    for shot in shots:
        x1, y1, x2, y2 = shot.split()
        lines.append(
            f"{delimiter}{x1}{delimiter} er for {delimiter}{y1}{delimiter} som {delimiter}{x2}{delimiter} er for {delimiter}{y2}{delimiter}"
        )
    return "\n".join(lines)


def create_default_prompt(masked: str, shots_prefix: str, delimiter: str) -> str:
    x1, y1, x2, y2 = masked.split()
    s = f"{delimiter}{x1}{delimiter} er for {delimiter}{y1}{delimiter} som {delimiter}{x2}{delimiter} er for {delimiter}"
    if shots_prefix:
        s = shots_prefix + "\n" + s
    return s


def main():
    """Measure generative language models' ability to finish sentences on the form 'X1 is to Y1 what X2 is to' in Norwegian."""
    parser = ArgumentParser()
    parser.add_argument(
        "--delimiter",
        "-d",
        help="Special token to pad around words for generation",
        default="#",
    )
    parser.add_argument(
        "--grammar_only",
        action="store_true",
        help="Will only use grammar analogies if flagged",
        default=False,
    )
    parser.add_argument(
        "--n_shots",
        "-n",
        type=int,
        help="Number of examples to show model before test example",
        default=0,
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for n shot example creation", default=42
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, help="Batch size for inference", default=100
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        help="Number of analogies to use for testing (maximum 17807)",
        default=0,
    )

    parser.add_argument(
        "--model_id",
        "-id",
        type=str,
        required=True,
        help="Huggingface hub model_id or path of model to evaluate",
    )
    parser.add_argument(
        "--quantize_bits",
        "-qb",
        type=int,
        help="Will load quantized model if set",
        choices=[4, 8],
    )
    parser.add_argument(
        "--hf_token",
        "-t",
        type=str,
        help="Huggingface hub token (for private models)",
        default=None,
    )
    parser.add_argument(
        "--eos_token", help="Custom token to stop generation at", default="#"
    )
    parser.add_argument(
        "--tokenizer_params",
        nargs="+",
        default=[],
        help="Parameters to the tokenize function (sequence of key=value pairs)",
    )
    parser.add_argument(
        "--generation_params",
        nargs="+",
        default=[],
        help="GenerationConfig parameters (sequence of key=value pairs)",
    )
    parser.add_argument(
        "--output_dir", "-o", type=Path, required=True, help="directory to save outputs"
    )

    args = parser.parse_args()

    if args.quantize_bits:
        model, tokenizer = get_quantized_model_and_tokenizer(
            args.model_id, bits=args.quantize_bits, token=args.hf_token
        )
    else:
        model, tokenizer = get_model_and_tokenizer(args.model_id, token=args.hf_token)

    if args.eos_token:
        args.eos_token = args.eos_token.replace("\\n", "\n").replace("\\t", "\t")
        model.generation_config.eos_token = tokenizer(args.eos_token).input_ids[0]
    else:
        args.eos_token = tokenizer.eos_token

    if args.generation_params:
        model_gen_config = model.generation_config.to_dict()
        model_gen_config.update(arglist_to_kwarg_dict(args.generation_params))
        model.generation_config = GenerationConfig.from_dict(model_gen_config)

    args.tokenizer_params = arglist_to_kwarg_dict(args.tokenizer_params)

    ds = load_dataset(only_grammar=args.grammar_only, num_examples=args.num_examples)
    shots = get_n_shots(dataset=ds, n=args.n_shots, seed=args.seed)
    shots_prefix = format_shots(shots, delimiter=args.delimiter)

    flat_ds = [(key, each) for key, list_ in ds.items() for each in list_]
    prompts = [
        create_default_prompt(
            shots_prefix=shots_prefix, masked=each, delimiter=args.delimiter
        )
        for _, each in flat_ds
    ]
    batches = [
        prompts[i : i + args.batch_size]
        for i in range(0, len(prompts), args.batch_size)
    ]
    generated_texts = [
        batch_generate(
            model=model,
            tokenizer=tokenizer,
            generation_config=model.generation_config,
            texts=batch,
            tokenizer_params=args.tokenizer_params,
        )
        for batch in batches
    ]
    generated_texts = [text for text_list in generated_texts for text in text_list]

    predicted_tokens = [
        get_first_predicted_word_only(
            prompt=model_prompt, generated_text=generated_text
        )
        for model_prompt, generated_text in zip(prompts, generated_texts)
    ]

    df_data = defaultdict(list)

    df_data["category"], _ = zip(*flat_ds)
    df_data["prompt"] = prompts
    df_data["generated_text"] = generated_texts
    df_data["predicted_token"] = predicted_tokens
    df_data["target_token"] = [each.split(" ")[-1] for _, each in flat_ds]
    df_data["correct_prediction"] = [
        predicted_token == target
        for predicted_token, target in zip(predicted_tokens, df_data["target_token"])
    ]

    output_dir = args.output_dir / "analogies_evaluation/"
    output_dir = get_output_dir(output_dir)

    with open(output_dir / "args.json", "w+") as f:
        json.dump(vars(args), f, default=str, indent=4, sort_keys=True)

    with open(output_dir / "generation_config.json", "w+") as f:
        json.dump(model.generation_config.to_dict(), f, indent=4, sort_keys=True)

    pd.DataFrame(df_data).to_csv(output_dir / "generated_text.csv", index=False)
