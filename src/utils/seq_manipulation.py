"""Functions to manipulate sequences, e.g. packing/padding"""
import torch
import torch.nn as nn
from typing import List


def pad_sequence(sequences, require_padding_mask=False, require_lens=False,
                 batch_first=False):
    """List of sequences to padded sequences

    Args:
        sequences: List of sequences (N, D)
        require_padding_mask:

    Returns:
        (padded_sequence, padding_mask), where
           padded sequence has shape (N_max, B, D)
           padding_mask will be none if require_padding_mask is False
    """
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    padding_mask = None
    padding_lens = None

    if require_padding_mask:
        B = len(sequences)
        seq_lens = list(map(len, sequences))
        padding_mask = torch.zeros((B, padded.shape[0]), dtype=torch.bool, device=padded.device)
        for i, l in enumerate(seq_lens):
            padding_mask[i, l:] = True

    if require_lens:
        padding_lens = [seq.shape[0] for seq in sequences]

    return padded, padding_mask, padding_lens


def unpad_sequences(padded, seq_lens):
    """Reverse of pad_sequence"""
    sequences = [padded[..., :seq_lens[b], b, :] for b in range(len(seq_lens))]
    return sequences


def split_src_tgt(feats, stack_lengths, dim=0):
    if isinstance(stack_lengths, torch.Tensor):
        stack_lengths = stack_lengths.tolist()

    B = len(stack_lengths) // 2
    separate = torch.split(feats, stack_lengths, dim=dim)
    return separate[:B], separate[B:]


def split_src_tgt_self(feats, stack_lengths, dim=0):
    if isinstance(stack_lengths, torch.Tensor):
        stack_lengths = stack_lengths.tolist()

    # 获取stack_lengths的一半长度
    half_len = len(stack_lengths) // 2

    # 使用stack_lengths将feats分割
    separate = torch.split(feats, stack_lengths, dim=dim)

    # 根据stack_lengths的前一半和后一半重新组合数据
    combined = [torch.cat([separate[i], separate[i + half_len]], dim=dim) for i in range(half_len)]

    return combined

def combine_src_tgt(combined_list: List[torch.Tensor], stack_lengths: List[int]) -> torch.Tensor:
    if isinstance(stack_lengths, torch.Tensor):
        stack_lengths = stack_lengths.tolist()

    half_len = len(stack_lengths) // 2

    src_tensors = []
    tgt_tensors = []
    for i in range(half_len):
        # 根据stack_lengths将每个combined tensor分割成两部分
        src_part, tgt_part = torch.split(combined_list[i], [stack_lengths[i], stack_lengths[i + half_len]], dim=0)
        src_tensors.append(src_part)
        tgt_tensors.append(tgt_part)

    # 首先连接所有的src_part，然后连接所有的tgt_part
    return torch.cat(src_tensors + tgt_tensors, dim=0)
