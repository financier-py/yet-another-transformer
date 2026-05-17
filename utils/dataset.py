import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.tokenizer import BPETokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.pairs = []
        for src_text, tgt_text in zip(src_texts, tgt_texts):
            src_ids = self.src_tokenizer.encode(src_text)
            tgt_ids = self.tgt_tokenizer.encode(tgt_text)

            sos_id = tgt_tokenizer.vocab["<SOS>"]
            eos_id = tgt_tokenizer.vocab["<EOS>"]
            tgt_ids = [sos_id] + tgt_ids + [eos_id]

            self.pairs.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def get_collate_fn(src_pad_idx: int, tgt_pad_idx: int):
    def collate_fn(batch):
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]

        src_padded = pad_sequence(
            src_batch, batch_first=True, padding_value=src_pad_idx
        )
        tgt_padded = pad_sequence(
            tgt_batch, batch_first=True, padding_value=tgt_pad_idx
        )

        return src_padded, tgt_padded

    return collate_fn
