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
from models.transformer.geo_module import RPEAttentionLayer, TransformerLayer

class TransformerCrossEncoder_self(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False,use_geo=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        # self.norm = norm
        self.return_intermediate = return_intermediate
        self.use_geo = use_geo

    def forward(self, src, tgt,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,):

        src_intermediate, tgt_intermediate = [], []

        if self.use_geo:
            for layer in self.layers:
                src, tgt = layer(src,
                                tgt,
                                src_pos,
                                tgt_pos,
                                src_mask,
                                tgt_mask)
                if self.return_intermediate:
                    src_intermediate.append(src)
                    tgt_intermediate.append(tgt)
                    
        # else:
        #     for layer in self.layers:
        #         src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
        #                         src_key_padding_mask=src_key_padding_mask,
        #                         tgt_key_padding_mask=tgt_key_padding_mask,
        #                         src_pos=src_pos, tgt_pos=tgt_pos)
        #         if self.return_intermediate:
        #             src_intermediate.append(self.norm(src) if self.norm is not None else src)
        #             tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        # if self.norm is not None:
        #     src = self.norm(src)
        #     tgt = self.norm(tgt)
        #     if self.return_intermediate:
        #         if len(self.layers) > 0:
        #             src_intermediate.pop()
        #             tgt_intermediate.pop()
        #         src_intermediate.append(src)
        #         tgt_intermediate.append(tgt)

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


class TransformerCrossEncoderLayer_self(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,
                 activation_fn='ReLU',
                 attention_type='dot_prod'
                 ):
        super().__init__()

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = RPEAttentionLayer(d_model, nhead, dropout=dropout, activation_fn=activation_fn)
            self.cross_attn = TransformerLayer(d_model, nhead, dropout=dropout, activation_fn=activation_fn)
        else:
            raise NotImplementedError

       

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    
    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        # 自注意力
        feats0, scores0, pos0 = self.self_attn(feats0, feats0, embeddings0, masks0)
        feats1, scores1, pos1 = self.self_attn(feats1, feats1, embeddings1, masks1)
        
        # 交叉注意力
        feats0, scores0 = self.cross_attn(feats0, feats1, pos0, pos1, memory_masks=masks1)
        feats1, scores1 = self.cross_attn(feats1, feats0, pos1, pos0, memory_masks=masks0)
        
        return feats0, feats1


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

