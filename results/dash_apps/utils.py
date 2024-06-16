import pandas as pd
from pathlib import Path

model_groups = {
    "base_scratch_and_deltas": [
        'mimir-mistral-7b-base-scratch','mimir-7b-factual','mimir-7b-nonfiction',
        'mimir-7b-untranslated-withnewspapers', 'mimir-7b-translated', 'mimir-7b-books',
        'mimir-7b-rightholders', 'mimir-7b-fiction','mimir-7b-untranslated', 'mimir-7b-newspapers',
    ],
    "base_and_extended": ['mimir-mistral-7b-base-scratch', 'mimir-mistral-7b-extended-scratch', 
                          'mimir-mistral-7b-base', 'mimir-mistral-7b-extended',
    ],
    "bases_and_intructs": [
        'mimir-mistral-7b-base-scratch', 'mimir-mistral-7b-base-scratch-instruct',
        'mimir-mistral-7b-extended-scratch', 'mimir-mistral-7b-extended-scratch-instruct',
        'mimir-mistral-7b-base', 'mimir-mistral-7b-base-instruct',
        'mimir-mistral-7b-extended', 'mimir-mistral-7b-extended-instruct',
    ]
}

def scale_rows(row):
    if row.score == "number_of_tokens":
        row["value"] = row.value / 100
        row["score"] = "number_of_tokens/100"
    if row.score == "lix_score":
        row["value"] = row.value / 10
        row["score"] = "lix_score/10"
    return row

def build_gen_df(overwrite=False):
    from diversity import compression_ratio, ngram_diversity_score

    for output_path in [Path("../contrastive"), Path("../greedy")]:
        if not overwrite and Path(f"lex_lix_generated_{output_path.name}.csv").exists():
            continue

        p = output_path / "mimir-project"
        df_data = {"model": [], "score": [], "value": [], "language": []}

        for e in p.iterdir():
            model_name = e.name 

            # Regn ut og rydd
            gen_text_df = pd.read_csv(e / "evaluate_all/generated_text.csv")    
            lex_div_df = pd.read_csv(e / "evaluate_all/lexical_diversity/scores_per_text.csv")
            if "language" not in lex_div_df.columns:
                lex_div_df["language"] = gen_text_df.language
                lex_div_df.to_csv(e / "evaluate_all/lexical_diversity/scores_per_text.csv", index=False)
            if "compression_ratio" not in lex_div_df.columns:
                lex_div_df["compression_ratio"] = [compression_ratio([text]) for text in gen_text_df["generated_text"]]
                lex_div_df["pos_compression_ratio"]  = [compression_ratio([text]) for text in lex_div_df["parts_of_speech"]]
                lex_div_df.to_csv(e / "evaluate_all/lexical_diversity/scores_per_text.csv", index=False)
            if "3_gram_diversity_score" not in lex_div_df.columns:
                lex_div_df["3_gram_diversity_score"] = gen_text_df["generated_text"].apply(lambda x: ngram_diversity_score([x], 3))
                lex_div_df["4_gram_diversity_score"] = gen_text_df["generated_text"].apply(lambda x: ngram_diversity_score([x], 4))
                lex_div_df.to_csv(e / "evaluate_all/lexical_diversity/scores_per_text.csv", index=False)

            lex_div_df["str_len"] = lex_div_df.text_str_len 
            lex_div_df = lex_div_df.drop(columns=["text_str_len"])

            desc = lex_div_df.describe()
            # desc = desc[["compression_ratio", "pos_compression_ratio", "self_bleu", "number_of_tokens", "str_len"]]
            for x in desc:
                df_data["model"].append(model_name)
                df_data["score"].append(x)
                df_data["value"].append(desc[x]["mean"])
                df_data["language"].append("both")
            
            for lang, df_ in lex_div_df.groupby("language"):
                desc = df_.describe()
                # desc = desc[["compression_ratio", "pos_compression_ratio", "self_bleu", "number_of_tokens", "str_len"]]
                for x in desc:
                    df_data["model"].append(model_name)
                    df_data["score"].append(x)
                    df_data["value"].append(desc[x]["mean"])
                    df_data["language"].append(lang)

            lix_df =  pd.read_csv(e / "evaluate_all/readability/scores_per_text.csv")
            if "language" not in lix_df.columns:
                lix_df["language"] = gen_text_df.language

            desc = lix_df.describe()
            for x in desc:
                df_data["model"].append(model_name)
                df_data["score"].append(x)
                df_data["value"].append(desc[x]["mean"])
                df_data["language"].append("both")
            
            for lang, df_ in lix_df.groupby("language"):
                desc = df_.describe()
                for x in desc:
                    df_data["model"].append(model_name)
                    df_data["score"].append(x)
                    df_data["value"].append(desc[x]["mean"])
                    df_data["language"].append(lang)


        df = pd.DataFrame(df_data)
        df.to_csv(f"lex_lix_generated_{output_path.name}.csv", index=False)


def build_source_df(overwrite=False):
    if not overwrite and Path("lex_lix_source.csv").exists():
        return
    df_data = {"delta": [], "language": [], "score": [], "value": []}
    for score_dir in [Path("../source_texts/lexical_diversity/"), Path("../source_texts/readability/")]:
        df = pd.read_csv(score_dir / "scores_per_text.csv")

        if "text_str_len" in df:
            df["str_len"] = df.text_str_len 
            df = df.drop(columns=["text_str_len"])

        desc = df.describe()
        for x in desc:
            df_data["delta"].append("all")
            df_data["language"].append("both")
            df_data["value"].append(desc[x]["mean"])
            df_data["score"].append(x)

        for e in score_dir.iterdir():
            if e.is_dir():
                if len(e.name) == 2:
                    langcode = e.name
                    df = pd.read_csv(e / "scores_per_text.csv")
                    if "text_str_len" in df:
                        df["str_len"] = df.text_str_len 
                        df = df.drop(columns=["text_str_len"])

                    desc = df.describe()
                    for x in desc:
                        df_data["delta"].append("all")
                        df_data["language"].append("nno" if langcode == "nn" else "nob")
                        df_data["value"].append(desc[x]["mean"])
                        df_data["score"].append(x)
                    for f in e.iterdir():
                        if f.is_dir():
                            dataset = f.name
                            df = pd.read_csv(f / "scores_per_text.csv")
                            if "text_str_len" in df:
                                df["str_len"] = df.text_str_len 
                                df = df.drop(columns=["text_str_len"])
                            desc = df.describe()
                            for x in desc:
                                df_data["delta"].append(dataset)
                                df_data["language"].append("nno" if langcode == "nn" else "nob")
                                df_data["value"].append(desc[x]["mean"])
                                df_data["score"].append(x)
                else:
                    dataset = e.name
                    df = pd.read_csv(e / "scores_per_text.csv")
                    if "text_str_len" in df:
                        df["str_len"] = df.text_str_len 
                        df = df.drop(columns=["text_str_len"])
                    desc = df.describe()
                    for x in desc:
                        df_data["delta"].append(dataset)
                        df_data["language"].append("both")
                        df_data["value"].append(desc[x]["mean"])
                        df_data["score"].append(x)
    df = pd.DataFrame(df_data)
    df.index = df.delta
    df.to_csv("lex_lix_source.csv", index=False)