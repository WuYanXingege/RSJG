from pathlib import Path

import torch

import ebjd
from ebjd.denoiser import (
    FactorizedFutureBlock,
    FactorizedJointAttentionBlock,
    GeometryConditionedJointDenoiser,
    JointEndpointBridgeDenoiser,
)
from ebjd.model import EBJDModel
from ebjd.representation import (
    EndpointBridgeDecomposition,
    EndpointBridgeRepresentation,
)
from ebjd.sampling import predict

from conftest import make_batch


def test_paper_names_are_compatibility_aliases_not_new_modules():
    assert FactorizedFutureBlock is FactorizedJointAttentionBlock
    assert JointEndpointBridgeDenoiser is GeometryConditionedJointDenoiser
    assert EndpointBridgeRepresentation is EndpointBridgeDecomposition
    model = EBJDModel()
    assert list(model._modules) == ["representation", "encoder", "denoiser"]
    assert len(model.state_dict()) == 307


def test_import_is_from_this_paper_workspace():
    root = Path(__file__).resolve().parents[1]
    assert root in Path(ebjd.__file__).resolve().parents


def test_full_raw_twenty_world_twenty_step_entry_is_finite_and_padding_safe():
    torch.manual_seed(41)
    model = EBJDModel().eval()
    batch = make_batch((1, 2), pixels=16)
    generator = torch.Generator(device="cpu").manual_seed(2035)
    with torch.no_grad():
        prediction, endpoint = predict(
            model, batch.observed, batch.semantic_maps, batch.valid,
            worlds=20, steps=20, generator=generator)
    assert prediction.shape == (2, 20, 2, 12, 2)
    assert endpoint.shape == (2, 20, 2, 2)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(endpoint).all()
    # FJABs mask padded slots and metrics exclude them. The decoded CV
    # baseline itself need not be literal zero, but must remain finite.
    assert torch.isfinite(prediction[0, :, 1]).all()
