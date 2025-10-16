import pytest
import torch

from modernvbert.models.modernvbert.modeling_modernvbert import ModernVBertForMaskedLM


@pytest.mark.parametrize("dt", [torch.float32, torch.float16])
def test_forward_pass(dt):
    model_id = "ModernVBERT/modernvbert"

    try:
        model = ModernVBertForMaskedLM.from_pretrained(model_id, torch_dtype=dt, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load pretrained model: {e}")

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    batch_size = 1
    seq_len = 8
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")

    logits = getattr(outputs, "logits", outputs[0])
    expected_vocab = vocab_size + getattr(model, "out_additional_features", 0)
    assert logits.shape == (batch_size, seq_len, expected_vocab)
    # model returns logits as float32 for stability
    assert logits.dtype == torch.float32
