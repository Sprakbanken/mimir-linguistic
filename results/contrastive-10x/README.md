This directory contains the results of running

```
python3 -m evaluate_all --model_id <model-id> --input_file sentence-starters/sentence_starters.csv --prompt_column prompt --lang_column language --output_dir results/contrastive-10x/<model_id> --seed <seed> --generation_params max_new_tokens=200 min_new_tokens=170 do_sample=True penalty_alpha=0.9 
```
for every <model_id> in the mimir project,  
for every <seed> from 40 to and including 49