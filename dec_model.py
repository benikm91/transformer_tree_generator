from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F

from experiments.tree.experiment_config import ModelConfig
from experiments.tree.positional_encoding import GlobalSinusoidalPositionalEncoding


def init_weights(model):
    def init_transformer_layer(layer):
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    nn.init.xavier_uniform_(model.token_embedding.weight)

    if isinstance(model.dec_pos_emb, nn.Embedding):
        nn.init.xavier_uniform_(model.dec_pos_emb.weight)

    for dec_layer in model.decoder:
        init_transformer_layer(dec_layer)

    nn.init.xavier_uniform_(model.lm_head.weight)


class Decoder(nn.Module):
    """ Standard Encoder Decoder Model """

    def __init__(self, vocab_size: int, dec_block_size: int, config: ModelConfig):
        super(Decoder, self).__init__()
        self.dec_block_size = dec_block_size

        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)

        if config.positional_encoding == 'global_learn':
            print("Using learnable global positional encoding")
            self.dec_pos_emb = nn.Embedding(dec_block_size, config.n_embd)  # learnable global PE
        elif config.positional_encoding == 'global_sinusoidal':
            print("Using fix global sinusoidal positional encoding")
            self.dec_pos_emb = GlobalSinusoidalPositionalEncoding(config.n_embd, dec_block_size)  # fix global PE
        else:
            raise ValueError(f'Unknown positional encoding: {config.positional_encoding}')

        self.decoder = nn.ModuleList([
            # Use torch's encoder layer (decoder requires memory) but with causal masking
            nn.TransformerEncoderLayer(d_model=config.n_embd, nhead=config.n_head, dim_feedforward=config.n_embd * 4, dropout=config.attn_pdrop, norm_first=True, activation='gelu')
            for _ in range(config.n_layer)
        ])

        self.drop = nn.Dropout(config.embd_pdrop)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        if config.tie_embeddings:
            print("Tying embeddings")
            self.lm_head.weight = self.token_embedding.weight
        else:
            print("Not tying embeddings")

        self.register_buffer('tgt_mask', self._create_causal_mask(dec_block_size))
        self.register_buffer('dec_pos', torch.arange(dec_block_size).unsqueeze(0))

        if config.weight_init:
            print("INITIALIZING WEIGHTS")
            init_weights(self)
        else:
            print("NOT INITIALIZING WEIGHTS")

    @staticmethod
    def _create_causal_mask(size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward_decoder(self, dec_x):
        dec_pos = self.dec_pos_emb(self.dec_pos[:, :dec_x.size(1)])
        dec_x_emb = self.drop(self.token_embedding(dec_x) + dec_pos)
        dec_x_emb = dec_x_emb.transpose(0, 1)  # TransformerDecoder expects shape (S, N, E)
        dec_output = dec_x_emb
        tgt_mask = self.tgt_mask[:dec_x.size(1), :dec_x.size(1)]

        for layer in self.decoder:
            dec_output = layer(dec_output, src_mask=tgt_mask)
        dec_output = dec_output.transpose(0, 1)  # back to (N, S, E)
        return dec_output

    def forward_logits(self, dec_output):
        logits = self.lm_head(dec_output)
        return logits

    def forward(self, dec_x, dec_y=None):
        dec_output = self.forward_decoder(dec_x)
        logits = self.forward_logits(dec_output)

        loss = None
        if dec_y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), dec_y.view(-1), ignore_index=-1)

        return logits, loss, loss

    def generate_sample(self, *, device, bos_id: int, eos_id: int) -> List[int]:
        sequence = [bos_id]
        dec_output = None
        for _ in range(self.dec_block_size):
            dec_output = self.forward_decoder(torch.tensor([sequence], device=device))
            logits = self.forward_logits(dec_output)
            next_token = torch.multinomial(torch.softmax(logits[0, -1], dim=-1), 1).item()
            sequence.append(next_token)
            if next_token == eos_id:
                break
        return sequence[1:-1]
