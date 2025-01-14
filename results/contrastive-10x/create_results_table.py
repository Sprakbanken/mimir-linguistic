import glob
import jsonlines
import csv
import pandas as pd

results = {}
with open("results.csv", "w") as wf:
    csvwriter = csv.writer(wf)
    csvwriter.writerow(["model", "run", "task", "value"])
    for file in glob.glob("mimir-project/*/*/results.jsonl"):
        model_name = file.split('/')[1]
        run = file.split('/')[2]
        with open(file) as f:
            json = jsonlines.jsonlines.Reader(f)
            for line in json:
                if line["dataset"] == "lexical_diversity":
                    tokens_nob = line["results"][0]["number_of_tokens_nob"]
                    tokens_nno = line["results"][0]["number_of_tokens_nno"]
                    cr_nob = line["results"][0]["compression_ratio_nob"]
                    cr_nno = line["results"][0]["compression_ratio_nno"]
                    sb_nob = line["results"][0]["self_bleu_nob"]
                    sb_nno = line["results"][0]["self_bleu_nno"]
                    csvwriter.writerow([model_name, run, "number_of_tokens_nob", tokens_nob])
                    csvwriter.writerow([model_name, run, "number_of_tokens_nno", tokens_nno])
                    csvwriter.writerow([model_name, run, "compression_ratio_nob", cr_nob])
                    csvwriter.writerow([model_name, run, "compression_ratio_nno", cr_nno])
                    csvwriter.writerow([model_name, run, "self_bleu_nob", sb_nob])
                    csvwriter.writerow([model_name, run, "self_bleu_nno", sb_nno])
                elif line["dataset"] == "readability":
                    lix_nob = line["results"][0]["lix_score_nob"]
                    lix_nno = line["results"][0]["lix_score_nno"]
                    csvwriter.writerow([model_name, run, "lix_score_nob", lix_nob])
                    csvwriter.writerow([model_name, run, "lix_score_nno", lix_nno])

results_table = pd.read_csv("results.csv")
results_table_pivot = results_table.pivot(index=["model", "run"], columns="task", values="value")
results_table_pivot = results_table_pivot.reset_index().groupby(by=["model"])[["compression_ratio_nob", "compression_ratio_nno", "lix_score_nob", "lix_score_nno", "self_bleu_nob", "self_bleu_nno"]].mean().reset_index()
results_table_pivot.to_csv("results_pivot.csv")
