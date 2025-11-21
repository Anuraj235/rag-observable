import os
import pandas as pd

CSV_PATH = "data_science_concepts.csv"  # adjust if needed
OUTPUT_DIR = "data"  # this should be the same as RAGPipeline.data_dir

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Basic sanity check
    if "Question" not in df.columns or "Answer" not in df.columns:
        raise ValueError("CSV must have 'Question' and 'Answer' columns")

    for idx, row in df.iterrows():
        q = str(row["Question"]).strip()
        a = str(row["Answer"]).strip()

        if not q and not a:
            continue

        filename = f"data_science_{idx:04d}.txt"
        out_path = os.path.join(OUTPUT_DIR, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {q}\n\n")
            f.write("Answer:\n")
            f.write(a)
            f.write("\n")

    print(f"Done! Wrote {len(df)} text files into '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
