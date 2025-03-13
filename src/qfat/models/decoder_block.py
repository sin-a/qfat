import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from qfat.conf.configs import DecoderBlockCfg

logger = logging.getLogger(__name__)


class DecoderBlock(nn.Module):
    """A Transformer decoder block.

    Inspired from https://github.com/karpathy/minGPT.
    """

    def __init__(self, cfg: DecoderBlockCfg):
        super().__init__()
        mha_cfg = cfg.mha_cfg
        self.ln_1 = nn.LayerNorm(mha_cfg.embed_dim)
        self.attn = nn.MultiheadAttention(**mha_cfg, batch_first=True)
        self.ln_2 = nn.LayerNorm(mha_cfg.embed_dim)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(
                    mha_cfg.embed_dim, cfg.mlp_expansion_factor * mha_cfg.embed_dim
                ),
                c_proj=nn.Linear(
                    cfg.mlp_expansion_factor * mha_cfg.embed_dim, mha_cfg.embed_dim
                ),
                act=nn.GELU(),
                dropout=nn.Dropout(cfg.residual_dropout),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x: torch.Tensor, **mha_kwargs: Dict[str, Any]) -> torch.Tensor:
        """Runs the input through a decoder block.

        The input is assumed to be of shape (batch_size, sequence_length, embedding_dim)
        The following sequence of operations is applied:
            1. A layer normalization is applied to the input (along the last dimension).
            2. Multiheaded attention which is computed across the heads and the output
                is then concatenated. The query, key and values are the same as the input
                x.
            3. The output of the attention layer is added to the original input.
            4. A layer norm is applied and the output is passed through a final mlp.
            5. Another residual connection is applied.

        Args:
            x (torch.tensor): The input sequence of shape (batch_size, sequence_length, embedding_dim).
            mha_kwargs (Dict[str, Any]): kwargs to the multi-headed-attention forward pass, e.g a mask.
        Returns:
            torch.Tensor: A contextualized sequence of shape (batch_size, sequence_length, embedding_dim).
        """
        if mha_kwargs == {}:
            logger.debug(
                "No arguments passed to the MultiHeadAttention layer forward pass."
            )
        x = self.ln_1(x)
        attn_out, _ = self.attn(query=x, key=x, value=x, **mha_kwargs)
        x = x + attn_out
        x = x + self.mlpf(self.ln_2(x))
        return x
