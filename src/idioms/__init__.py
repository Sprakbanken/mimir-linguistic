from argparse import ArgumentParser
from collections import defaultdict
import json
import pandas as pd
from pathlib import Path
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


def load_masked_idioms(filename: str) -> dict[str, list[str]]:
    """Create dictionary of idiom_start : list[acceptable_last_words]"""
    with open(filename) as f:
        lines = f.readlines()
    masked_idioms = defaultdict(list)
    for l in lines:
        words = l.strip().split(" ")
        masked_idioms[" ".join(words[:-1])].append(words[-1])
    return masked_idioms


def format_prompt(prompt: str, idiom_start: str) -> str:
    return f"{prompt}{idiom_start}"


def main():
    """Measure generative language models' ability to finish Norwegian idioms.
    Will present idioms with the last word missing, and count how often the model correctly generates the last word of the idiom.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--prompt",
        "-p",
        help="Custom prompt to include before idiom",
        default="Idiom:\n",
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
        "--eos_token", help="Custom token to stop generation at", default="\n"
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

    nno_idioms = load_masked_idioms(
        "src/idioms/norwegian-idioms/nno_idioms_curated.txt"
    )
    nob_idioms = load_masked_idioms(
        "src/idioms/norwegian-idioms/nob_idioms_curated.txt"
    )

    df_data = defaultdict(list)

    formatted_prompts = [
        format_prompt(args.prompt, idiom_start) for idiom_start in nno_idioms
    ] + [format_prompt(args.prompt, idiom_start) for idiom_start in nob_idioms]

    generated_texts = batch_generate(
        model=model,
        tokenizer=tokenizer,
        generation_config=model.generation_config,
        texts=formatted_prompts,
        tokenizer_params=args.tokenizer_params,
    )
    predicted_tokens = [
        get_first_predicted_word_only(
            prompt=model_prompt, generated_text=generated_text
        )
        for model_prompt, generated_text in zip(formatted_prompts, generated_texts)
    ]

    df_data["prompt"] = formatted_prompts
    df_data["generated_text"] = generated_texts
    df_data["predicted_token"] = predicted_tokens
    df_data["accepted_predictions"] = list(nno_idioms.values()) + list(
        nob_idioms.values()
    )
    df_data["correct_prediction"] = [
        predicted_token in accepted_predictions
        for predicted_token, accepted_predictions in zip(
            df_data["predicted_token"], df_data["accepted_predictions"]
        )
    ]
    df_data["language"] = ["nno"] * len(nno_idioms) + ["nob"] * len(nob_idioms)

    output_dir = args.output_dir / "idioms_evaluation/"
    output_dir = get_output_dir(output_dir)

    with open(output_dir / "args.json", "w+") as f:
        json.dump(vars(args), f, default=str, indent=4, sort_keys=True)

    with open(output_dir / "generation_config.json", "w+") as f:
        json.dump(model.generation_config.to_dict(), f, indent=4, sort_keys=True)

    pd.DataFrame(df_data).to_csv(output_dir / "generated_text.csv", index=False)
