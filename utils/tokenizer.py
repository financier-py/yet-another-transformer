from collections import defaultdict
import re


class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.vocab = {}
        self.merges = {}

    def _get_stats(self, corpus_cnts: dict):
        pairs = defaultdict(int)
        for word, freq in corpus_cnts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_vocab(self, pair, corpus_counts):
        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        replacement = "".join(pair)
        for word in corpus_counts:
            w_out = p.sub(replacement, word)
            v_out[w_out] = corpus_counts[word]
        return v_out

    def train(self, texts: list[str]):
        corpus_counts = defaultdict(int)
        for text in texts:
            words = text.strip().split()
            for word in words:
                char_word = " ".join(list(word)) + " </w>"
                corpus_counts[char_word] += 1

        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        cur_id = len(self.vocab)

        for word in corpus_counts.keys():
            for token in word.split():
                if token not in self.vocab:
                    self.vocab[token] = cur_id
                    cur_id += 1

        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pairs = self._get_stats(corpus_counts)
            if not pairs:
                break

            best_pair = max(pairs, key=lambda k: pairs[k])
            new_token = "".join(best_pair)
            
            self.merges[best_pair] = new_token
            if new_token not in self.vocab:
                self.vocab[new_token] = cur_id
                cur_id += 1
                
            corpus_counts = self._merge_vocab(best_pair, corpus_counts)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str):
        words = text.strip().split()
        encoded_tokens = []

        for word in words:
            w_chars = " ".join(list(word)) + " </w>"

            for pair, merged in self.merges.items():
                bigram = re.escape(" ".join(pair))
                p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
                w_chars = p.sub(merged, w_chars)

            for subword in w_chars.split():
                token_id = self.vocab.get(subword, self.vocab["<UNK>"])
                encoded_tokens.append(token_id)
        return encoded_tokens

    def decode(self, ids: list[int]):
        tokens = [self.inverse_vocab.get(i, "<UNK>") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()