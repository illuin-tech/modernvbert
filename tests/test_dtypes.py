import pytest
import torch

from modernvbert.models.modernvbert.modeling_modernvbert import ModernVBertForMaskedLM

@pytest.mark.parametrize("dt", [torch.float32, torch.float16])
def test_model_parameter_dtypes(dt):
    model_id = "ModernVBERT/modernvbert"

    try:
        model = ModernVBertForMaskedLM.from_pretrained(
            model_id, torch_dtype=dt, trust_remote_code=True
        )
    except Exception as e:
        pytest.skip(f"Could not load pretrained model: {e}")

    # assert all params have expected dtype
    mismatches = [
        (name, param.dtype)
        for name, param in model.named_parameters()
        if param.dtype != dt
    ]

    assert not mismatches, f"Found parameter dtype mismatches: {mismatches[:10]}"
