import torch


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
    """
    src: (batch_size, src_len)
    tgt: (batch_size, tgt_len)
    """

    # (batch_size, 1, 1, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    tgt_len = tgt.size(1)
    look_ahead_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
    tgt_mask = tgt_pad_mask & look_ahead_mask

    return src_mask, tgt_mask