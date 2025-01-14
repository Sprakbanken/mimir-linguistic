# Linguistic evaluation of generative language models

This repo contains functionality for linguistic evaluation of generative language models.

## Setup
You will need Python 3.10:

```bash
python -m venv <venv_name>      # make a virtual environment

. <venv_name>/bin/activate      # activate the environment

pip install .                   # install dependencies and modules
```


# How to run full pipeline
To run all metrics on a specific model, you can use the `evaluate_all` module. See the below sections for more detailed explanations of the different metrics.

Run `python3 -m evaluate_all -h` for argument information.

## Required arguments
- model_id:                  huggingface hub model id or path to local model that can be loaded with transformers.AutoModelForCausalLM
- output_dir:                directory to store all outputs of evaluation pipeline
- input_file:                .csv file with prompts for text generation
- prompt_column:             prompt column in input_file


## Optional arguments
- lang_column:              language column in input file
- hf_token:                 huggingface token with read access (private models)                
- quantize_bits:            4 or 8. Will load quantized model if set                            
- eos_token:                 custom eos_token                                          
- tokenizer_params:          sequence of key=value pairs for the tokenizer*           
- generation_params:         sequence of key=value pairs for the generation config*  
- seed:                      random seed for generation and other sampling
- batch_size:                number of examples per batch (for the tasks that may require batching)  
- texts_per_prompt:          number of texts to generate for each prompt
- ns:                        list of Ns for n-gram diversity score

\* Depending on the model, some values of generation_params and/or tokenizer_params must be set for the generation to work properly  
See [this document](https://huggingface.co/docs/transformers/v4.15.0/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for paramaters that can be used with  `--tokenizer_params`  
See [this document](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/text_generation#transformers.GenerationConfig) for paramaters that can be used with `--generation_params`

## Example run 
```bash
python3 -m evaluate_all --model_id mimir-project/nb-llama-1.5b-mimirbase --input_file sentence-starters/sentence_starters.csv --prompt_column prompt --output_dir output/mimir-project/nb-llama-1.5b-mimirbase --generation_params max_new_tokens=200 min_new_tokens=170 --plot_scores
```

## Output
Will create the following directory structure in `output_dir`:
```
evaluate_all/
├─ lexical_diversity/
│  ├─ scores_across_texts.csv
│  ├─ scores_per_text.csv
├─ readability/
│  ├─ scores_across_texts.csv
│  ├─ scores_per_text.csv
├─ synonyms/
│  ├─ generated_text.csv
args.json
generated_text.csv
generation_config.json
results.jsonl
```

`results.jsonl`  contains overall results for all tasks in the evaluate_all pipeline

# Lexical diversity module

The following metrics may be used with the module "lexical_diviersities". For evaluation, metrics on generated text should be compared with the same metrics on training material within the same domain.

- Type-token ratio
- Moving average type-token ratio
- Compression ratio
- Part-of-speech compression ratio   
- Stop word density
- SELF-BLEU

## How to use

```bash
python3 -m lexical_diversities  --input_file <texts.csv>                                                
                                --text_column <column name in the CSV where the texts can be found>     
                                --ns    <list of Ns for N-gram diversity score>                                  
                                --output_dir    <output directory>
```

Generates the following files:
- `<output_dir>/lexical_diversity/scores_per_text.csv`
- `<output_dir>/lexical_diversity/scores_across_texts.csv`

# Readability module
Measures the readability of generated text. Currently implemented:
- Lix score

## How to use

```bash
python3 -m readability  --input_file <texts.csv>                                                
                                --text_column <column name in the CSV where the texts can be found>
                                --output_dir    <output directory>
```

Generates the following files:
- `<output_dir>/readability/scores_per_text.csv`
- `<output_dir>/readability/scores_across_texts.csv`

# Generation utility module

In order to generate text for evaluation, use the generation module.  
It loads a model with the transformers library and a .csv file with prompts to generate from.  

Required paramaters:
```
--input_file        <.csv file with prompts>
--prompt_column     <prompt column in input_file>
--texts_per_prompt  <number of texts to generate for each prompt>
--seed              <random seed for text generation>
--model_id          <HuggingFace model id or local path>      
--output_dir        <output directory>        
```
Optional parameters:
```
--hf_token          <hf token with read access (private models)>                
--quantize_bits     <4 or 8 bit, model quantization>                            
--eos_token         <custom eos_token>                                          
--tokenizer_params  <sequence with key=value pairs for the tokenizer>           
--generation_params <sequence with key=value pairs for the generation config>   
```
See [this document](https://huggingface.co/docs/transformers/v4.15.0/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for paramaters that can be used with  `--tokenizer_params`  
See [this document](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/text_generation#transformers.GenerationConfig) for paramaters that can be used with `--generation_params`

Generates the following files:
- `<output_dir>/<evaluation_name>/args.json`           
- `<output_dir>/<evaluation_name>/generated_text.csv`         
- `<output_dir>/<evaluation_name>/generation_config.json` 

