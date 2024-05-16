from argparse import ArgumentParser
from collections import defaultdict
import json
import pandas as pd
from pathlib import Path
import random
from transformers import GenerationConfig
from string import punctuation
from utils import (
    get_model_and_tokenizer,
    get_quantized_model_and_tokenizer,
    arglist_to_kwarg_dict,
    batch_generate,
    get_output_dir,
    get_first_predicted_word_only,
)


def load_synonym_groups() -> list[list[str]]:
    with open("src/synonyms/norwegian-synonyms/norwegian-synonyms-grouped.json") as f:
        return [sorted({k, *l}) for k, v in json.loads(f.read()).items() for l in v]


def create_examples(
    synonym_groups: list[list[str]],
    seed: int,
    num_list_examples: int,
    num_total_examples: int,
) -> list[tuple[str, list[str]]]:
    random.seed(seed)
    random.shuffle(synonym_groups)
    synonym_groups = [e for e in synonym_groups if len(e) >= num_list_examples]
    examples = []
    for i, syn_list in enumerate(synonym_groups):
        if i == num_total_examples:
            break
        new_list = random.sample(syn_list, k=num_list_examples)
        odd_one_out = random.choice(synonym_groups[i - 20])
        while odd_one_out in new_list:
            odd_one_out = random.choice(
                synonym_groups[random.randint(0, len(synonym_groups))]
            )

        new_list.append(odd_one_out)
        random.shuffle(new_list)
        examples.append((odd_one_out, new_list))
    return examples


def format_example(
    prompt_pre: str, prompt_post: str, example: tuple[str, list[str]]
) -> str:
    return f"{prompt_pre}{example[1]}{prompt_post}"


def main():
    """Measure generative language models' ability to pick the odd word out in a list of synonyms and one unrelated word.
    Will present a list of known synonyms with one random word in the mix, and count how often the model correctly generates the odd word out.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--prompt_pre",
        "-p",
        help="Custom prompt to include before example",
        default="Hvilket ord hører ikke hjemme i lista?:\n",
    )
    parser.add_argument(
        "--prompt_post",
        "-pp",
        help="Custom prompt to include after example",
        default="\nOrdet som ikke hører hjemme er:",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="Random seed for creating synonym test examples",
        default=42,
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, help="Batch size for inference", default=100
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        help="Number of synonym lists to use for testing (maximum 33909)",
        default=1000,
    )
    parser.add_argument(
        "--num_list_examples",
        type=int,
        help="Number of synonyms to use in each example",
        default=2,
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

    groups = load_synonym_groups()
    examples = create_examples(
        synonym_groups=groups,
        seed=args.seed,
        num_total_examples=args.num_examples,
        num_list_examples=args.num_list_examples,
    )
    formatted_prompts = [
        format_example(
            prompt_pre=args.prompt_pre, prompt_post=args.prompt_post, example=ex
        )
        for ex in examples
    ]

    batches = [
        formatted_prompts[i : i + args.batch_size]
        for i in range(0, len(formatted_prompts), args.batch_size)
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
        for model_prompt, generated_text in zip(formatted_prompts, generated_texts)
    ]

    df_data = defaultdict(list)
    df_data["prompt"] = formatted_prompts
    df_data["generated_text"] = generated_texts
    df_data["predicted_token"] = predicted_tokens
    df_data["target_token"] = [target for target, _ in examples]
    df_data["correct_prediction"] = [
        predicted_token == target
        for predicted_token, (target, _) in zip(predicted_tokens, examples)
    ]

    output_dir = args.output_dir / "synonyms_evaluation/"
    output_dir = get_output_dir(output_dir)

    with open(output_dir / "args.json", "w+") as f:
        json.dump(vars(args), f, default=str, indent=4, sort_keys=True)

    with open(output_dir / "generation_config.json", "w+") as f:
        json.dump(model.generation_config.to_dict(), f, indent=4, sort_keys=True)

    pd.DataFrame(df_data).to_csv(output_dir / "generated_text.csv", index=False)
