from argparse import ArgumentParser
import json
import pandas as pd
from pathlib import Path
from transformers import GenerationConfig, set_seed
from mimir_linguistic.lexical_diversities import calculate_lexical_diversity_scores
from mimir_linguistic.readability import calculate_lix_scores
import torch
from mimir_linguistic.utils import (
    get_model_and_tokenizer,
    get_quantized_model_and_tokenizer,
    arglist_to_kwarg_dict,
    batch_generate,
    get_output_dir,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=Path,
        help=".csv-file with prompts for text generation",
        required=True,
    )
    parser.add_argument(
        "--prompt_column",
        "-pc",
        help="Column in input file containing prompts",
        required=True,
    )
    parser.add_argument(
        "--lang_column",
        required=False,
        default="",
        help="Column in input file containing language (optional)",
    )
    parser.add_argument(
        "--texts_per_prompt",
        "-tpp",
        type=int,
        default=1,
        help="Number of texts to generate per prompt",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        help="directory to save outputs of evaluation",
        required=True,
    )
    parser.add_argument(
        "--model_id",
        help="huggingface hub model id or path to local model",
        required=True,
    )
    parser.add_argument(
        "--hf_token", help="huggingface token with read access", required=False
    )
    parser.add_argument(
        "--quantize_bits",
        "-qb",
        type=int,
        help="Will load quantized model if set",
        choices=[4, 8],
    )
    parser.add_argument(
        "--eos_token", help="Custom token to stop generation at", default=None
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
        "--batch_size", type=int, help="Batch size for inference", default=100
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling and generation"
    )
    return parser


def main():
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = get_parser()
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

    if args.texts_per_prompt > 1:
        model.generation_config.do_sample = True
        model.generation_config.num_return_sequences = args.texts_per_prompt

    if args.generation_params:
        model_gen_config = model.generation_config.to_dict()
        model_gen_config.update(arglist_to_kwarg_dict(args.generation_params))
        model.generation_config = GenerationConfig.from_dict(model_gen_config)

    args.tokenizer_params = arglist_to_kwarg_dict(args.tokenizer_params)

    output_dir = get_output_dir(args.output_dir / "evaluate_all/")

    with open(output_dir / "args.json", "w+") as f:
        json.dump(
            vars(args), f, default=str, indent=4, sort_keys=True, ensure_ascii=False
        )

    with open(output_dir / "generation_config.json", "w+") as f:
        json.dump(
            model.generation_config.to_dict(),
            f,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
        )

    # STEP 1: Generate text
    prompt_df = pd.read_csv(args.input_file)
    prompts = prompt_df[args.prompt_column].to_list()

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

    df_data = {
        "prompt": [p for prompt in prompts for p in [prompt] * args.texts_per_prompt],
        "generated_text": [text for text_list in generated_texts for text in text_list],
    }
    if args.lang_column:
        df_data["language"] = [
            l
            for lang in prompt_df[args.lang_column]
            for l in [lang] * args.texts_per_prompt
        ]

    pd.DataFrame(df_data).to_csv(output_dir / "generated_text.csv", index=False)
    torch.cuda.empty_cache()

    # STEP 2: Calculate lexical diversity and readability scores on the generated text
    df = pd.read_csv(output_dir / "generated_text.csv")

    scores_per_text, scores_across_texts = calculate_lexical_diversity_scores(
        df=df,
        text_column="generated_text",
    )

    lex_output_dir = get_output_dir(output_dir / "lexical_diversity/")
    scores_per_text.to_csv(lex_output_dir / "scores_per_text.csv", index=False)
    scores_across_texts.to_csv(lex_output_dir / "scores_across_texts.csv", index=False)

    lix_output_dir = get_output_dir(output_dir / "readability/")
    scores_per_text, scores_across_texts = calculate_lix_scores(
        df=df,
        text_column="generated_text",
    )
    scores_per_text.to_csv(lix_output_dir / "scores_per_text.csv", index=False)
    scores_across_texts.to_csv(lix_output_dir / "scores_across_texts.csv", index=False)

    if args.lang_column:
        for lang, df_ in df.groupby("language"):
            output_lang_dir = get_output_dir(output_dir / "lexical_diversity" / lang)
            df_.index = range(len(df_))
            scores_per_text, scores_across_texts = calculate_lexical_diversity_scores(
                df=df_,
                text_column="generated_text",
            )
            scores_per_text.to_csv(output_lang_dir / "scores_per_text.csv", index=False)
            scores_across_texts.to_csv(
                output_lang_dir / "scores_across_texts.csv", index=False
            )

            output_lang_dir = get_output_dir(output_dir / "readability" / lang)

            scores_per_text, scores_across_texts = calculate_lix_scores(
                df=df_,
                text_column="generated_text",
            )
            scores_per_text.to_csv(output_lang_dir / "scores_per_text.csv", index=False)
            scores_across_texts.to_csv(
                output_lang_dir / "scores_across_texts.csv", index=False
            )

    df = pd.read_csv(output_dir / "lexical_diversity/scores_across_texts.csv")
    lex_div_scores = {k: v[0] for k, v in df.to_dict().items()}

    lex_dir = output_dir / "lexical_diversity"
    for e in lex_dir.iterdir():
        if e.is_dir():
            langcode = e.name
            df = pd.read_csv(e / "scores_across_texts.csv")
            for k, v in df.to_dict().items():
                lex_div_scores[f"{k}_{langcode}"] = v[0]

    df = pd.read_csv(output_dir / "readability/scores_across_texts.csv")
    lix_scores = {"lix_score": df.lix_score[0]}

    lix_dir = output_dir / "readability"
    for e in lix_dir.iterdir():
        if e.is_dir():
            langcode = e.name
            df = pd.read_csv(e / "scores_across_texts.csv")
            lix_scores[f"lix_score_{langcode}"] = df.lix_score[0]

    dicts = [
        {
            "dataset": "lexical_diversity",
            "model": args.model_id,
            "results": [lex_div_scores],
        },
        {"dataset": "readability", "model": args.model_id, "results": [lix_scores]},
    ]
    df = pd.DataFrame(dicts)
    df.to_json(output_dir / "results.jsonl", lines=True, orient="records")
