from pathlib import Path 
import pandas as pd

if __name__ == "__main__":
    results_dir = Path('results')

    for file in results_dir.glob('**/*.jsonl'):
        print(file.name)
        df = pd.read_json(file, lines=True)
        if "dataset" not in df.columns:
            print("Dataset column not found in", file)

        df = df[df['dataset'].isin(['lexical_diversity', 'readability'])]
        df.to_json(file, orient='records', lines=True)
