from pathlib import Path
from string import punctuation
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
import torch
from typing import Optional


def get_model_and_tokenizer(
    model_id: str, token: Optional[str] = None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    return model, tokenizer


def get_quantized_model_and_tokenizer(
    model_id: str, bits=8, token: Optional[str] = None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    match bits:
        case 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        case 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        case _:
            print("Parameter bits must be in [4, 8]")
            return (None, None)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        token=token,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    return model, tokenizer


def str_to_type(str_: str) -> str | float | bool:
    if str_.isnumeric():
        return float(str_)
    if str_.lower() == "false":
        return False
    if str_.lower() == "true":
        return True
    return str_


def arglist_to_kwarg_dict(args: list[str]) -> dict[str, str | float | bool]:
    arg_dict = {}
    if all(["=" in arg for arg in args]):
        for e in args:
            parameter, value = e.split("=")
            arg_dict[parameter] = str_to_type(value)
    elif len(args) % 2 == 0:
        for i, e in enumerate(args):
            if i % 2:
                continue
            parameter = e
            value = e[i + 1]
            arg_dict[parameter] = str_to_type(value)
    else:
        raise Exception(
            "Malformatted kwarg parameters. Provide list of 'param=value' or alternating param value pairs"
        )
    return arg_dict


def get_first_predicted_word_only(prompt: str, generated_text: str) -> str:
    return generated_text[len(prompt) :].strip().split(" ")[0].strip(punctuation)


def get_output_dir(output_dir: Path) -> Path:
    """Make a new output_dir if it already exists"""
    if output_dir.exists():
        num_dirs = len(
            [
                e
                for e in output_dir.parent.iterdir()
                if e.is_dir() and output_dir.name in e.name
            ]
        )
        output_dir = output_dir.parent / f"{output_dir.name}_{num_dirs}"
    output_dir.mkdir(parents=True)
    return output_dir


@torch.no_grad
def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: GenerationConfig,
    texts: list[str],
    tokenizer_params: dict = {},
) -> list[str]:
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True, **tokenizer_params).to(
        "cuda"
    )
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


@torch.no_grad
def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: GenerationConfig,
    text: str,
    tokenizer_params: dict = {},
) -> str:
    inputs = tokenizer(text, return_tensors="pt", **tokenizer_params).to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
