"""Modified from DETR's transformer.py

- Cross encoder layer is similar to the decoder layers in Transformer, but
  updates both source and target features
- Added argument to control whether value has position embedding or not for
  TransformerEncoderLayer and TransformerDecoderLayer
- Decoder layer now keeps track of attention weights
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerCrossEncoder(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False,use_geo=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.use_geo = use_geo

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        src_intermediate, tgt_intermediate = [], []

        if self.use_geo:
            for layer in self.layers:
                tgt_intermediate, src_intermediate = layer(tgt,
                                src,
                                tgt_mask,
                                src_mask,
                                tgt_key_padding_mask,
                                src_key_padding_mask)
                # if self.return_intermediate:
                #     src_intermediate.append(self.norm(src_mask) if self.norm is not None else src_mask)
                #     tgt_intermediate.append(self.norm(tgt_mask) if self.norm is not None else tgt_mask)
                if self.norm is not None:
                    src = self.norm(src_mask)
                    tgt = self.norm(tgt_mask)
                    if self.return_intermediate:
                        if len(self.layers) > 0:
                            src_intermediate.pop()
                            tgt_intermediate.pop()
                        src_intermediate.append(src)
                        tgt_intermediate.append(tgt)

                if self.return_intermediate:
                    return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

                return src.unsqueeze(0), tgt.unsqueeze(0)
                    
        else:
            for layer in self.layers:
                src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                src_pos=src_pos, tgt_pos=tgt_pos)
                if self.return_intermediate:
                    src_intermediate.append(self.norm(src) if self.norm is not None else src)
                    tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

                if self.norm is not None:
                    src = self.norm(src)
                    tgt = self.norm(tgt)
                    if self.return_intermediate:
                        if len(self.layers) > 0:
                            src_intermediate.pop()
                            tgt_intermediate.pop()
                        src_intermediate.append(src)
                        tgt_intermediate.append(tgt)

                if self.return_intermediate:
                    return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

                return src.unsqueeze(0), tgt.unsqueeze(0)

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        
        # output, attn_weights = self.self_attn(query, key, value, 
                                    #  key_padding_mask=key_padding_mask, 
                                    #  attn_mask=attn_mask)
        # query, key, value: 这些都是你要输入的主要张量。在自注意力的情境下，它们通常都是同一个张量，但在其他情况（如编码器-解码器注意力）中，它们可能是不同的。
        # 可选， 如果不给，则默认所有位置参与计算
        # key_padding_mask: 一个布尔掩码，用于指定哪些条目不应被考虑（例如，因为它们是填充的）。
        # 可选， 如果不给，则默认所有位置参与计算
        # attn_mask: 一个掩码，用于阻止某些位置参与注意力计算。例如，在解码器中，为了确保输出位置只注意前面的位置，你可能会使用这个。
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        # 做残差
        src = src + self.dropout1(src2)
        # 在normal一下
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention
        
        # LayerNorm
        src2 = self.norm1(src)
        # 点云特征+位置编码
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        # 得到q,k
        q = k = src2_w_pos
        
        # output, attn_weights = self.self_attn(query, key, value, 
                                    #  key_padding_mask=key_padding_mask, 
                                    #  attn_mask=attn_mask)
        # output: 经过多头注意力处理后的张量。
        # attn_weights: 表示注意力权重的张量，可用于分析或可视化。
        
        # query, key, value: 这些都是你要输入的主要张量。在自注意力的情境下，它们通常都是同一个张量，但在其他情况（如编码器-解码器注意力）中，它们可能是不同的。
        # 可选， 如果不给，则默认所有位置参与计算
        # key_padding_mask: 一个布尔掩码，用于指定哪些条目不应被考虑（例如，因为它们是填充的）。
        # 可选， 如果不给，则默认所有位置参与计算
        # attn_mask: 一个掩码，用于阻止某些位置参与注意力计算。例如，在解码器中，为了确保输出位置只注意前面的位置，你可能会使用这个。
        src2, satt_weights_s = self.self_attn(q, k,
                                              # 自注意力 所以q,k,v全部一样
                                              value=src2_w_pos if self.sa_val_has_pos_emb else src2,
                                              # 哪些是padding上去的特征
                                              attn_mask=src_mask,
                                              # 似乎没传递？None？
                                              key_padding_mask=src_key_padding_mask)
        # 做残差 A+result(A)
        src = src + self.dropout1(src2)
        
        # 同上

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              # v也是q,k一样的？
                                              value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
                                              # 
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        # Cross attention
        # LayerNorm
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        # 添加位置编码
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        # Q是scr， K，V是tgt
        src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        # Q是tgt， K, V是src
        tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)
        # 进行dropout
        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        
        # LayerNorm
        src2 = self.norm3(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # A + result(A)
        src = src + self.dropout3(src2)

        # LayerNorm
        tgt2 = self.norm3(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # A + result(A)
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        # 可视化注意力分析
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

