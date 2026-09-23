import torch
import torch.nn as nn


class EncoderDecoderTransformer(nn.Module):
    """Transformer encoder-decoder para generar respuestas a partir de contexto y pregunta."""

    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        max_src_len=512,
        max_tgt_len=64,
        pad_token_id=0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.src_position = nn.Embedding(max_src_len, d_model)
        self.tgt_position = nn.Embedding(max_tgt_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.output = nn.Linear(d_model, vocab_size)

    def _add_positions(self, tokens, embedding, position_embedding):
        sequence_length = tokens.size(1)
        positions = torch.arange(sequence_length, device=tokens.device).unsqueeze(0)
        return embedding(tokens) * (self.d_model ** 0.5) + position_embedding(positions)

    def forward(self, source, target_input):
        source_padding_mask = source.eq(self.pad_token_id)
        target_padding_mask = target_input.eq(self.pad_token_id)
        target_mask = nn.Transformer.generate_square_subsequent_mask(
            target_input.size(1), device=target_input.device
        )

        source_embeddings = self._add_positions(
            source, self.src_embedding, self.src_position
        )
        target_embeddings = self._add_positions(
            target_input, self.tgt_embedding, self.tgt_position
        )
        hidden_states = self.transformer(
            source_embeddings,
            target_embeddings,
            tgt_mask=target_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output(hidden_states)