# Linguistic evaluation of generative language models

This repo contains functionality for linguistic evaluation of generative language models.

## Setup
You will need Python 3.10:

```bash
python -m venv <venv_name>      # make a virtual environment

. <venv_name>/bin/activate      # activate the environment

pip install .                   # install dependencies and modules
```

# Metrics for use on generated text

The following metrics may be used with the module "lexical_diviersities". For evaluation, metrics on generated text should be compared with the same metrics on training material within the same domain.

## Lexical diversity
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

## Readability
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

# Metrics on tasks related to language use and world knowledge
All modules in this category load models with transformers.AutoModelForCausalLM  
Shared module paramaters are as follows:

Required paramaters:
```
--model_id      <HuggingFace hub model id or path to local model>      
--output_dir    <output directory>        
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

Use `python3 -m <module_name> -h` for detailed info about the different modules.

## Analogies 
Based on [The Norwegian Analogy Test Set](https://github.com/ltgoslo/norwegian-analogies) from LTG.  
It measures the models' accuracy in completing analogy pairs (〈a:b, c:d, a is to b, as c is to d).  
We observed that it helped to pad the words in each analogy pair with special delimiter tokens (such as hashes, so that the prompts look like: "#X1# er for #Y2# som #X2# er for #")

How to run:

```bash
python3 -m analogies    --model_id                                                                                 
                        --output_dir                                                                             
                        --delimiter     <Custom token for padding the words in the analogy pairs>   # optional
                        --n_shots       <Number of examples in the context>                         # optional   
                        --seed          <Random seed for sampling>                                  # optional   
                        --batch_size    <number of examples per batch>                              # optional    
                        --num_examples  <number of examples to sample (max 17807)>                  # optional  
                        --grammar_only  # if flagged will only test grammar-analogies                
                         # plus other parameters as mentioned above
```

Run `python3 -m analogies -h` for detailed information.

## Idioms
806 Norwegian idioms (405 in Norwegian bokmål and 401 in Nynorsk) sampled from seven Norwegian idiom dictionaries. Measures the models' accuracy in completing the last word of an idiom.

How to run:

```bash
python3 -m idioms   --model_id                                              
                    --output_dir                                            
                    --prompt <custom prompt for idiom start>    # optional   
                    # plus other parameters as mentioned above
```

Run `python3 -m idioms -h` for detailed information.

## Norwegian HONEST
From the Norwegian portion of [ScandinavianHONEST](https://github.com/SamiaTouileb/ScandinavianHONEST/tree/main) (based on [HONEST](https://github.com/MilaNLProc/honest?tab=readme-ov-file)).  
Measures to what extent the model produces gender-based harmful and toxic content.  

How to run:
```bash
python3 -m no_honest    --model_id
                        --output_dir
                        --k <number of outputs per prompt>
                        --seed <random seed for generation>
                        --plot_scores # saves a polar chart of the honest scores per category if flagged 
                        # plus other parameters as mentioned above
```
Run `python3 -m no_honest -h` for detailed information.

Creates `<output_dir>/honest_evaluation/honest_scores.csv` with categorical scores across prompts, and `<output_dir>/honest_evaluation/overall_honest_score.txt` with the overall honest score. The overall HONEST score is the average score across classes. Read more about the score in the [original paper](https://aclanthology.org/2021.naacl-main.191/)



## Synonyms
Based on [The Norwegian Synonymy Test Set](https://github.com/ltgoslo/norwegian-synonyms) from LTG. Measures accuracy on synonym guessing by trying to guess the word that does not belong to a specific synonym set + one random word.

How to run:

```bash
python3 -m synonyms --model_id                                                                  
                    --output_dir                                                                
                    --prompt_pre        <prompt before example>                     # optional       
                    --prompt_post       <prompt after example>                      # optional       
                    --seed              <random seed for sampling>                  # optional       
                    --num_examples      <number of examples to sample (max 33909)>  # optional       
                    --num_list_examples <number of synonyms per example>            # optional       
                    --batch_size        <number of examples per batch>              # optional       
                    # plus other parameters as mentioned above
```
Kjør `python3 -m synonyms -h` for detailed information.

