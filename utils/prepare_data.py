import re
import json
import pandas as pd
from pathlib import Path

from utils.tokenizer import BPETokenizer
from config import CSV_PATH, TOKENIZER_PATH, VOCAB_SIZE


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


def save_tokenizer(tokenizer: BPETokenizer, file_path: Path):
    json_merges = {f"{k[0]},{k[1]}": v for k, v in tokenizer.merges.items()}
    data = {"vocab": tokenizer.vocab, "merges": json_merges}

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_tokenizer(vocab_size: int, file_path: Path):
    with open(file_path / "bpe_tokenizer.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = BPETokenizer(vocab_size)
    tokenizer.vocab = data["vocab"]

    tokenizer.merges = {}
    for k, v in data["merges"].items():
        pair = tuple(k.split(","))
        tokenizer.merges[pair] = v

    tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    return tokenizer


if __name__ == "__main__":
    src_data, tgt_data = prepare_pairs(CSV_PATH)
    full_corpus = src_data + tgt_data

    print("Примеры получившихся пар")
    for i in range(3):
        print(f"src: {src_data[i]}")
        print(f"tgt: {tgt_data[i]}")
        print("-" * 30)

    tokenizer = BPETokenizer(VOCAB_SIZE)
    tokenizer.train(full_corpus)

    save_tokenizer(tokenizer, TOKENIZER_PATH)
