# README
This directory contains the outputs of running:

```bash
python3 -m evaluate_all --model_id <model_id> \
--input_file sentence-starters/sentence_starters.csv \
--prompt_column prompt \
--lang_column language \
--output_dir results/greedy/<model_id> \
--generation_params max_new_tokens=200 min_new_tokens=170 \
```

for every <model_id> in the mimir project.  

