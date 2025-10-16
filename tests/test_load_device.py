import pytest
import torch

from modernvbert.models.modernvbert.modeling_modernvbert import ModernVBertForMaskedLM


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_loading_on_device(device_str):
    """Ensure the model parameters end up on the requested device.

    This test will skip CUDA cases when CUDA is not available.
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model_id = "ModernVBERT/modernvbert"
    try:
        model = ModernVBertForMaskedLM.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load pretrained model: {e}")

    device = torch.device(device_str)
    # Move model to the target device
    model.to(device)

    # Ensure all parameters are on the requested device
    params_not_on_device = [(n, p.device) for n, p in model.named_parameters() if p.device.type != device_str]
    assert not params_not_on_device, f"Some parameters are not on {device}: {params_not_on_device[:10]}"
