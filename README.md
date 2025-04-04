# Linguistic evaluation of generative language models (Mímir project)

This repo contains the code and results of linguistic evaluation of the models trained in the [Mímir project](https://www.ntnu.edu/norllm/the-project-mimir-on-copyrighted-content).  
See paper [here](https://dspace.ut.ee/items/a111af0f-9aaf-422b-a69f-6f25ce99f25d)  
See press release and report [here](https://www.nb.no/pressemeldinger/forskningsprosjekt-viser-rettighetsbelagt-innhold-gir-norske-sprakmodeller-hoy-kvalitet/) (Norwegian only) 

## Mímir results 
See the results of the linguistic evaluation in [results/](results)  

The results used for the technical report are [results/contrastive-10x/results_table.csv](results/contrastive-10x/results_table.csv)

See how the tables were created in [notebooks/create_results_tables.ipynb](notebooks/create_results_tables.ipynb)

## Setup
### With pdm
```bash
pdm install
```

### With venv
You will need Python <3.13>=3.10:
```bash
python -m venv <venv_name>      # make a virtual environment

. <venv_name>/bin/activate      # activate the environment

pip install .                   # install dependencies and modules
```

# How to run full pipeline
To run all metrics on a specific model, you can use the `mimir_linguistic.evaluate_all` module, or just the outer `mimir_linguistic` module.  
See the below sections for more detailed explanations of the different metrics.

Run `python3 -m mimir_linguistic -h` or `pdm run python -m mimir_linguistic -h` for argument information.

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

### With pdm
```bash
pdm run python -m mimir_linguistic --model_id mimir-project/nb-llama-1.5b-mimirbase --input_file sentence-starters/sentence_starters.csv --prompt_column prompt --output_dir output/mimir-project/nb-llama-1.5b-mimirbase --generation_params max_new_tokens=200 min_new_tokens=170
```


### With venv
```bash
python3 -m mimir_linguistic --model_id mimir-project/nb-llama-1.5b-mimirbase --input_file sentence-starters/sentence_starters.csv --prompt_column prompt --output_dir output/mimir-project/nb-llama-1.5b-mimirbase --generation_params max_new_tokens=200 min_new_tokens=170
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
args.json
generated_text.csv
generation_config.json
results.jsonl
```

`results.jsonl`  contains overall results for all tasks in the `evaluate_all` pipeline

# Lexical diversity module

The following metrics may be used with the module `mimir_linguistic.lexical_diviersities`.  
For evaluation, metrics on generated text should be compared with the same metrics on training material within the same domain.

- Compression ratio
- SELF-BLEU

## How to use

`pdm run python -m mimir_linguistic.lexical_diversities` or `python3 -m mimir_linguistic.lexical_diversities`

### Arguments:
- input_file: a text .csv file with generated text                                                
- text_column: column name in the CSV where the generated text is
- output_dir: output directory

## Generates the following files:
- `<output_dir>/lexical_diversity/scores_per_text.csv`
- `<output_dir>/lexical_diversity/scores_across_texts.csv`

# Readability module
Measures the readability of generated text. 

Currently implemented:
- Lix score

## How to use
`pdm run python -m mimir_linguistic.readability` or `python3 -m mimir_linguistic.readability`

### Arguments:  
- input_file: a text .csv file with generated text                                                
- text_column: column name in the CSV where the generated text is
- output_dir:    output directory

## Generates the following files:
- `<output_dir>/readability/scores_per_text.csv`
- `<output_dir>/readability/scores_across_texts.csv`

# Generation utility module
In order to generate text for evaluation, use the generation module. 
It loads a model with the transformers library and a .csv file with prompts to generate from.  


## How to use:

`pdm run python -m mimir_linguistic.generate` or `python3 -m mimir_linguistic.generate`


### Required arguments:
- input_file: .csv file with prompts
- prompt_column: prompt column in input_file
- texts_per_prompt: number of texts to generate for each prompt
- seed: random seed for text generation
- model_id: HuggingFace model id or local path      
- output_dir: output directory

### Optional arguments:

- hf_token:           hf token with read access (private models)               
- quantize_bits:      4 or 8 bit, model quantization                            
- eos_token:          custom eos_token                                          
- tokenizer_params:   sequence with key=value pairs for the tokenizer    
- generation_params:  sequence with key=value pairs for the generation config

See [this document](https://huggingface.co/docs/transformers/v4.15.0/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for paramaters that can be used with  `--tokenizer_params`  
See [this document](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/text_generation#transformers.GenerationConfig) for paramaters that can be used with `--generation_params`

## Generates the following files:
- `<output_dir>/<evaluation_name>/args.json`           
- `<output_dir>/<evaluation_name>/generated_text.csv`         
- `<output_dir>/<evaluation_name>/generation_config.json` 

