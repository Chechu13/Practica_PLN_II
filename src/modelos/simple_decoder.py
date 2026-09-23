import torch
import torch.nn as nn


class SimpleDecoder(nn.Module):
    """Arquitectura de un Decoder Transformer desde cero."""

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.pos_encoder(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        out = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.fc_out(out)