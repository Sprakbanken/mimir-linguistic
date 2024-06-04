# README
This directory contains outputs of

```bash
python3 -m evaluate_all --model_id <model_id> \
--input_file sentence-starters/sentence_starters.csv \
--prompt_column prompt \
--lang_column language \
--output_dir output/<model_id> \
--generation_params max_new_tokens=200 min_new_tokens=170 \
--plot_scores
```