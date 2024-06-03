from argparse import ArgumentParser
from honest import honest
import json
import pandas as pd
from pathlib import Path
from transformers import GenerationConfig, set_seed
import unidecode
from utils import (
    get_model_and_tokenizer,
    get_quantized_model_and_tokenizer,
    batch_generate,
    arglist_to_kwarg_dict,
    get_output_dir,
)


def get_evaluator_and_templates() -> (
    tuple[honest.HonestEvaluator, dict[str, dict[str, str]]]
):
    ### from https://github.com/MilaNLProc/honest/blob/main/honest/honest.py
    ### from https://colab.research.google.com/drive/1vVgarK99TVm2QKdaJtJjye1470BD1_Bb?usp=sharing#scrollTo=G5NFAgaQwhP-

    evaluator = honest.HonestEvaluator("en")
    evaluator.language = "no"
    evaluator.hurtlex = pd.read_csv(
        f"https://raw.githubusercontent.com/MilaNLProc/hurtlex/master/lexica/{evaluator.language.upper()}/1.2/hurtlex_{evaluator.language.upper()}.tsv",
        sep="\t",
    )
    evaluator.hurtlex = evaluator.hurtlex[evaluator.hurtlex["level"] == "conservative"]
    evaluator.hurtlex["lemma"] = evaluator.hurtlex["lemma"].apply(
        lambda accented_string: unidecode.unidecode(str(accented_string))
    )

    evaluator.categories = set(evaluator.hurtlex["category"].unique())
    evaluator.words = set(evaluator.hurtlex["lemma"].unique())

    masked_templates = evaluator.templates(
        data_set="binary",
        path="src/no_honest/ScandinavianHONEST/resources/binary/no_template.tsv",
    )

    return evaluator, masked_templates


def remove_mask(text: str) -> str:
    return text.replace(" [M].", "")


def plot(honest_df: pd.DataFrame, img_path: Path, k: int) -> None:
    ### from https://colab.research.google.com/drive/1vVgarK99TVm2QKdaJtJjye1470BD1_Bb?usp=sharing#scrollTo=G5NFAgaQwhP-

    import plotly.graph_objects as go

    # Initialize selected lexicon categories
    hurtlex_cat = [
        "animals",
        "crime and immoral behavior",
        "derogatory words",
        "female genitalia",
        "male genitalia",
        "homosexuality",
        "potential negative connotations",
        "professions and occupations",
        "prostitution",
        "social and economic disadvantage",
    ]
    hurtlex_cat_names = ["an", "re", "cds", "asf", "asm", "om", "qas", "pa", "pr", "is"]

    df_identity = honest_df.groupby("category")[hurtlex_cat_names].sum()
    df_count = honest_df.groupby("category")[hurtlex_cat_names].count() * k
    df_perc = df_identity / df_count * 100

    plots = [
        go.Scatterpolar(r=df_perc.loc[i], theta=hurtlex_cat, fill="toself", name=i)
        for i, row in df_perc.iterrows()
    ]

    fig = go.Figure(
        data=plots,
        layout=go.Layout(
            polar={"radialaxis": {"visible": True}},
        ),
    )

    fig.write_image(img_path)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--k", "-k", type=int, help="Number of outputs for each example", default=5
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for generation", default=42
    )
    parser.add_argument("--plot_scores", action="store_true", default=False)
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
    return parser


def calculate_HONEST_score(
    model,
    tokenizer,
    tokenizer_params,
    k: int,
    batch_size: int,
    plot_scores: bool,
    output_dir: Path,
):
    evaluator, masked_templates = get_evaluator_and_templates()
    prompts_without_masks = [remove_mask(e) for e in masked_templates]

    batches = [
        prompts_without_masks[i : i + batch_size]
        for i in range(0, len(prompts_without_masks), batch_size)
    ]
    generated_texts = [
        batch_generate(
            model=model,
            tokenizer=tokenizer,
            generation_config=model.generation_config,
            texts=batch,
            tokenizer_params=tokenizer_params,
        )
        for batch in batches
    ]
    generated_texts = [text for text_list in generated_texts for text in text_list]

    only_generated_parts = [
        text[len(prompt) :]
        for text, prompt in zip(generated_texts, prompts_without_masks)
    ]

    # #Compute HONEST score
    honest_score, honest_df = evaluator.honest_dataframe(
        only_generated_parts, masked_templates
    )

    honest_df.to_csv(output_dir / "honest_scores.csv", index=False)

    with open(output_dir / "overall_honest_score.txt", "w+") as f:
        f.write(str(honest_score))

    df_data = {
        "prompt": [p for prompt in prompts_without_masks for p in [prompt] * k],
        "generated_text": generated_texts,
    }

    pd.DataFrame(df_data).to_csv(output_dir / "generated_text.csv", index=False)

    if plot_scores:
        plot(honest_df=honest_df, img_path=output_dir / "plot.png", k=k)


def main():
    args = get_parser().parse_args()
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

    model.generation_config.num_return_sequences = args.k
    model.generation_config.do_sample = True
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.max_new_tokens = 10

    if args.generation_params:
        model_gen_config = model.generation_config.to_dict()
        model_gen_config.update(arglist_to_kwarg_dict(args.generation_params))
        model.generation_config = GenerationConfig.from_dict(model_gen_config)

    model.generation_config.num_return_sequences = args.k

    args.tokenizer_params = arglist_to_kwarg_dict(args.tokenizer_params)

    output_dir = args.output_dir / "honest_evaluation"
    output_dir = get_output_dir(output_dir)

    with open(output_dir / "args.json", "w+") as f:
        json.dump(vars(args), f, default=str, indent=4, sort_keys=True)

    with open(output_dir / "generation_config.json", "w+") as f:
        json.dump(model.generation_config.to_dict(), f, indent=4, sort_keys=True)

    calculate_HONEST_score(
        model=model,
        tokenizer=tokenizer,
        tokenizer_params=args.tokenizer_params,
        k=args.k,
        plot_scores=args.plot_scores,
        output_dir=output_dir,
    )

    print(f"See results at {output_dir}")
