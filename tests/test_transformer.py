
import sys
from pathlib import Path

import pytest

# 실험 설치(requirements-experiments.txt)가 없으면 건너뛴다
torch = pytest.importorskip("torch")

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from train_transformer import FraudNeuralNetwork


def test_tabular_mlp_output_shape():
    model = FraudNeuralNetwork(
        tab_dim=15,
        seq_dim=9,
        use_sequence=False,
    )

    tabular = torch.randn(4, 15)
    sequence = torch.randn(4, 5, 9)
    mask = torch.ones(4, 5, dtype=torch.bool)

    output = model(tabular, sequence, mask)

    assert output.shape == (4,)
    assert torch.isfinite(output).all()


def test_transformer_output_shape():
    model = FraudNeuralNetwork(
        tab_dim=15,
        seq_dim=9,
        use_sequence=True,
    )

    tabular = torch.randn(4, 15)
    sequence = torch.randn(4, 5, 9)
    mask = torch.ones(4, 5, dtype=torch.bool)

    output = model(tabular, sequence, mask)

    assert output.shape == (4,)
    assert torch.isfinite(output).all()


def test_all_padding_is_safe():
    model = FraudNeuralNetwork(
        tab_dim=15,
        seq_dim=9,
        use_sequence=True,
    )

    model.eval()

    tabular = torch.randn(2, 15)
    sequence = torch.zeros(2, 5, 9)
    mask = torch.zeros(2, 5, dtype=torch.bool)

    with torch.no_grad():
        output = model(tabular, sequence, mask)

    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_masked_tokens_do_not_change_prediction():
    torch.manual_seed(42)

    model = FraudNeuralNetwork(
        tab_dim=15,
        seq_dim=9,
        use_sequence=True,
    )

    model.eval()

    tabular = torch.randn(2, 15)
    sequence = torch.randn(2, 5, 9)

    mask = torch.tensor([
        [False, False, False, True, True],
        [False, False, True, True, True],
    ])

    changed = sequence.clone()
    changed[~mask] = 999.0

    with torch.no_grad():
        original = model(tabular, sequence, mask)
        modified = model(tabular, changed, mask)

    assert torch.allclose(
        original, modified,
        atol=1e-5,
        rtol=1e-5,
    )