from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F

from experiments.tree.dec_model import init_weights
from experiments.tree.experiment_config import ModelConfig
from experiments.tree.positional_encoding import GlobalSinusoidalPositionalEncoding


class DecoderSplit(nn.Module):
    """ Standard Encoder Decoder Model """

    def __init__(self, vocab_size: int, dec_block_size: int, config: ModelConfig, loss_strategy: str='standard', branch_factor: int=None):
        super(DecoderSplit, self).__init__()
        self.dec_block_size = dec_block_size
        self.loss_strategy = loss_strategy
        self.branch_factor = branch_factor

        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)

        if config.positional_encoding == 'global_learn':
            self.dec_pos_emb = nn.Embedding(dec_block_size+1, config.n_embd)  # learnable global PE
        elif config.positional_encoding == 'global_sinusoidal':
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
            self.lm_head.weight = self.token_embedding.weight

        # Cache masks for reuse
        self.register_buffer('dec_i2i_mask', self._create_causal_mask(dec_block_size))
        self.register_buffer('dec_i2p_mask', torch.ones(dec_block_size, dec_block_size, dtype=torch.bool))
        self.register_buffer('dec_p2i_mask', self._create_causal_mask(dec_block_size))
        self.register_buffer('dec_p2p_mask', ~torch.eye(dec_block_size, dtype=torch.bool))

        self.register_buffer('dec_pos', torch.arange(dec_block_size+1).unsqueeze(0))

        if config.weight_init:
            init_weights(self)


    @staticmethod
    def _create_causal_mask(size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward_decoder(self, dec_x_input):
        dec_pos_input = self.dec_pos_emb(self.dec_pos[:, :dec_x_input.size(1)])  # drop last PE token (EOS)
        dec_pos_pred = self.dec_pos_emb(self.dec_pos[:, 1:(dec_x_input.size(1)+1)])  # drop first PE token (BOS)
        dec_x_input_emb = self.drop(self.token_embedding(dec_x_input) + dec_pos_input)
        dec_x_pred = torch.zeros_like(dec_x_input_emb) + dec_pos_pred

        # stack along sequnence dimension dec_x_input_emb and dec_x_pred
        dec_x_emb = torch.cat([dec_x_input_emb, dec_x_pred], dim=1)
        # dec_x_input_emb causal self-attend
        # Create attention masks
        seq_len_input = dec_x_input.size(1)
        seq_len_pred = dec_x_pred.size(1)

        # Correctly slice masks
        dec_i2i_mask = self.dec_i2i_mask[:seq_len_input, :seq_len_input]
        dec_i2p_mask = self.dec_i2p_mask[:seq_len_input, :seq_len_pred]
        dec_p2i_mask = self.dec_p2i_mask[:seq_len_pred, :seq_len_input]
        dec_p2p_mask = self.dec_p2p_mask[:seq_len_pred, :seq_len_pred]

        # Combine masks into a single tgt_mask
        tgt_mask_top = torch.cat([dec_i2i_mask, dec_i2p_mask], dim=1)
        tgt_mask_bottom = torch.cat([dec_p2i_mask, dec_p2p_mask], dim=1)
        tgt_mask = torch.cat([tgt_mask_top, tgt_mask_bottom], dim=0)

        dec_output = dec_x_emb
        dec_output = dec_output.transpose(0, 1)  # TransformerDecoder expects shape (S, N, E)
        for layer in self.decoder:
            dec_output = layer(dec_output, src_mask=tgt_mask)
        dec_output = dec_output.transpose(0, 1)  # back to (N, S, E)
        dec_pred_input = dec_output[:, :dec_x_input.size(1), :]
        dec_pred_output = dec_output[:, dec_x_input.size(1):, :]
        return dec_pred_input, dec_pred_output

    def forward_logits(self, dec_output):
        logits = self.lm_head(dec_output)
        return logits

    def forward(self, dec_x, dec_y=None):
        dec_input_score, dec_pred_score = self.forward_decoder(dec_x)
        dec_input_logit = self.forward_logits(dec_input_score)
        dec_pred_logit = self.forward_logits(dec_pred_score)

        loss, loss_pred = None, None
        if dec_y is not None:
            loss_input = F.cross_entropy(dec_input_logit.view(-1, dec_input_logit.size(-1)), dec_x.view(-1), ignore_index=-1)
            if self.loss_strategy == 'standard':
                loss_pred = F.cross_entropy(dec_pred_logit.view(-1, dec_pred_logit.size(-1)), dec_y.view(-1), ignore_index=-1)
            elif self.loss_strategy == 'min':
                branch_factor = self.branch_factor
                root_loss = F.cross_entropy(dec_pred_logit[:, 0], dec_y[:, 0], ignore_index=-1)
                eos_loss = F.cross_entropy(dec_pred_logit[:, -1], dec_y[:, -1], ignore_index=-1)

                # Create permutations of dec_y by rolling
                relevant = dec_y[:, 1:-1]
                dec_y_grouped = torch.zeros(relevant.size(0), relevant.size(1), branch_factor, dtype=torch.long, device=dec_y.device)

                for i in range(branch_factor):
                    dec_y_grouped[:, :, i] = torch.roll(relevant, shifts=-i, dims=1)

                mask = torch.ones(branch_factor, branch_factor, device=dec_y.device)
                for i in range(1, branch_factor):
                    mask[i, i:] = 0

                stacked_mask = mask

                # broadcast stacked_mask to fit dec_y_grouped (N, S, branch_factor)
                N, S = dec_y_grouped.size(0), dec_y_grouped.size(1)
                stacked_mask = stacked_mask.repeat(S // branch_factor, 1)
                stacked_mask = stacked_mask.unsqueeze(0).repeat(N, 1, 1)  # Shape becomes (N, S, branch_factor)

                dec_y_grouped = torch.where(stacked_mask == 1, dec_y_grouped, -1)

                dec_pred_logit_grouped = dec_pred_logit[:, 1:-1]
                dec_pred_logit_grouped = dec_pred_logit_grouped.unsqueeze(2).repeat(1, 1, branch_factor, 1)

                # Reshape logits and targets to apply cross_entropy
                logits_reshaped = dec_pred_logit_grouped.view(-1, dec_pred_logit_grouped.size(-1))
                targets_reshaped = dec_y_grouped.view(-1)

                # Compute cross-entropy loss without reduction
                x = F.cross_entropy(logits_reshaped, targets_reshaped, reduction='none', ignore_index=-1)
                x = x.view(N, S, branch_factor)
                # replace stack_mask with infinity
                high_value = float('inf')
                x = torch.where(stacked_mask == 1, x, high_value)
                x = x.min(dim=-1).values
                tree_body_loss = torch.mean(x)

                # Calculate the mean loss across all positions and batches
                loss_pred = root_loss + tree_body_loss + eos_loss

            loss = loss_input + loss_pred

        return dec_pred_logit, loss, loss_pred

    def generate_sample(self, *, device, bos_id: int, eos_id: int) -> List[int]:
        sequence = [bos_id]
        for _ in range(self.dec_block_size):
            pred_logits = self.forward_logits(
                self.forward_decoder(torch.tensor([sequence], device=device))[1]
            )
            next_token = torch.multinomial(torch.softmax(pred_logits[0, -1], dim=-1), 1).item()
            sequence.append(next_token)
            if next_token == eos_id:
                break
        return sequence[1:-1]
