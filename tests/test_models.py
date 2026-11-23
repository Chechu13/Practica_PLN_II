import pytest
import torch

from src.modelos.encoder_decoder import EncoderDecoderTransformer
from src.modelos.simple_decoder import SimpleDecoder

@pytest.mark.parametrize("kwargs", [
    {}, {"d_model": 32, "nhead": 4, "num_layers": 1},
    {"d_model": 64, "nhead": 8, "num_layers": 2},
])


def test_simple_decoder_output_shape(kwargs):
    model = SimpleDecoder(vocab_size=50, **kwargs)
    assert model(torch.randint(0, 50, (2, 7))).shape == (2, 7, 50)


def test_simple_decoder_batch_one_single_token():
    model = SimpleDecoder(vocab_size=20, d_model=16, nhead=4, num_layers=1)
    assert model(torch.randint(0, 20, (1, 1))).shape == (1, 1, 20)


def test_simple_decoder_empty_sequence_is_supported():
    model = SimpleDecoder(vocab_size=20, d_model=16, nhead=4, num_layers=1)
    output = model(torch.empty((1, 0), dtype=torch.long))
    assert output.shape == (1, 0, 20)


def test_simple_decoder_max_position_limit():
    model = SimpleDecoder(vocab_size=20, d_model=16, nhead=4, num_layers=1, max_seq_len=3)
    with pytest.raises((IndexError, RuntimeError)):
        model(torch.randint(0, 20, (1, 4)))


def build_encoder_decoder(**kwargs):
    return EncoderDecoderTransformer(
        vocab_size=40, d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=64, max_src_len=12, max_tgt_len=8, **kwargs,
    )


def test_encoder_decoder_output_shape():
    model = build_encoder_decoder()
    assert model(torch.randint(1, 40, (2, 6)), torch.randint(1, 40, (2, 4))).shape == (2, 4, 40)


def test_encoder_decoder_padding_masks():
    model = build_encoder_decoder()
    source = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    target = torch.tensor([[1, 2, 0], [3, 4, 0]])
    output = model(source, target)
    assert output.shape == (2, 3, 40)
    assert torch.isfinite(output).all()


def test_encoder_decoder_batch_one_single_token():
    model = build_encoder_decoder()
    assert model(torch.tensor([[1]]), torch.tensor([[2]])).shape == (1, 1, 40)

@pytest.mark.parametrize(("source_len", "target_len"), [(13, 1), (2, 9)])


def test_encoder_decoder_position_limits(source_len, target_len):
    model = build_encoder_decoder()
    source = torch.ones((1, source_len), dtype=torch.long)
    target = torch.ones((1, target_len), dtype=torch.long)
    with pytest.raises((IndexError, RuntimeError)):
        model(source, target)


def test_encoder_decoder_custom_padding_id():
    model = build_encoder_decoder(pad_token_id=1)
    assert model.pad_token_id == 1
    assert model(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 4]])).shape == (1, 2, 40)

@pytest.mark.parametrize("kwargs", [{"d_model": 30, "nhead": 4}, {"d_model": 32, "nhead": 0}])


def test_decoder_rejects_invalid_attention_dimensions(kwargs):
    with pytest.raises((AssertionError, ValueError)):
        SimpleDecoder(vocab_size=20, **kwargs)
