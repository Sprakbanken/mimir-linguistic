from argparse import ArgumentParser
import json
import pandas as pd
from pathlib import Path
from transformers import GenerationConfig, set_seed
from mimir_linguistic.utils import (
    get_model_and_tokenizer,
    get_quantized_model_and_tokenizer,
    batch_generate,
    arglist_to_kwarg_dict,
    get_output_dir,
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file", "-i", type=Path, help=".csv-file with prompts for generation"
    )
    parser.add_argument(
        "--prompt_column", "-pc", help="Column in input file containing prompts"
    )
    parser.add_argument(
        "--texts_per_prompt",
        "-tpp",
        type=int,
        default=1,
        help="Number of texts to generate per prompt",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for generation", default=42
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
    set_seed(args.seed)

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

    if args.texts_per_prompt > 1:
        model.generation_config.do_sample = True
    model.generation_config.num_return_sequences = args.texts_per_prompt

    args.tokenizer_params = arglist_to_kwarg_dict(args.tokenizer_params)

    df = pd.read_csv(args.input_file)
    prompts = df[args.prompt_column].to_list()

    generated_texts = batch_generate(
        texts=prompts,
        model=model,
        generation_config=model.generation_config,
        tokenizer=tokenizer,
        tokenizer_params=args.tokenizer_params,
    )

    output_dir = args.output_dir / "generate"
    output_dir = get_output_dir(output_dir)

    with open(output_dir / "args.json", "w+") as f:
        json.dump(vars(args), f, default=str, indent=4, sort_keys=True)

    with open(output_dir / "generation_config.json", "w+") as f:
        json.dump(model.generation_config.to_dict(), f, indent=4, sort_keys=True)

    df_data = {
        "prompt": [p for prompt in prompts for p in [prompt] * args.texts_per_prompt],
        "generated_text": generated_texts,
    }

    pd.DataFrame(df_data).to_csv(output_dir / "generated_text.csv", index=False)

    print(f"See results at {output_dir}")
