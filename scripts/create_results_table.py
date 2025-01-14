from pathlib import Path
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    results_dir = Path("results/contrastive-10x/")

    scores_to_keep = [
        "compression_ratio_nob",
        "compression_ratio_nno",
        "lix_score_nob",
        "lix_score_nno",
        "self_bleu_nob",
        "self_bleu_nno",
    ]

    linguistic_scores = defaultdict(list)

    for model_dir in results_dir.glob("*/*/"):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        score_lists = defaultdict(list)

        for result_file in model_dir.glob("*/*results.jsonl"):
            df = pd.read_json(result_file, lines=True)
            res = {}
            for tup in df.itertuples():
                res.update(tup.results[0])

            for score in scores_to_keep:
                score_lists[score].append(res[score])

        avg_scores = {k: sum(v) / len(v) for k, v in score_lists.items()}

        linguistic_scores["model"].append(model_name)
        for k, v in avg_scores.items():
            linguistic_scores[k].append(v)

    df = pd.DataFrame(linguistic_scores)
    df.to_csv(results_dir / "results_table.csv", index=False)

    results_dir = Path("results/greedy/results")

    linguistic_scores = defaultdict(list)
    for results_file in results_dir.iterdir():
        model_name = results_file.stem
        df = pd.read_json(results_file, lines=True)
        res = {}
        for tup in df.itertuples():
            res.update(tup.results[0])

        linguistic_scores["model"].append(model_name)
        for score in scores_to_keep:
            linguistic_scores[score].append(res[score])

    df = pd.DataFrame(linguistic_scores)
    df.to_csv(results_dir.parent / "results_table.csv", index=False)
