import re
import pandas as pd
from pathlib import Path


def clean_text(text: str):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[ ]+", " ", text)
    return text.strip()


def prepare_pairs(csv_path: Path):
    df = pd.read_csv(csv_path, usecols=["text"])
    df = df.dropna(subset=["text"])

    src_texts, tgt_texts = [], []
    for idx, row in df.iterrows():
        poem_text = clean_text(row["text"])
        lines = [line.strip() for line in poem_text.split("\n") if line.strip()]

        for i in range(0, len(lines) - 1, 2):
            src_line = lines[i]
            tgt_line = lines[i + 1]

            src_texts.append(src_line)
            tgt_texts.append(tgt_line)
    return src_texts, tgt_texts


CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "poems.csv"

if __name__ == "__main__":
    src_data, tgt_data = prepare_pairs(CSV_PATH)

    print("Примеры получившихся пар")
    for i in range(3):
        print(f"src: {src_data[i]}")
        print(f"tgt: {tgt_data[i]}")
        print("-" * 30)
