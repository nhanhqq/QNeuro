import numpy as np
import pytest
import torch

from qneuro_paperlite.features import FEATURE_NAMES, paperlite_features
from qneuro_paperlite.model import PaperLiteQNeuro, count_trainable_parameters


def test_feature_shape_finite_and_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 6, 512)).astype(np.float32)
    first = paperlite_features(x, fs=128, frame_seconds=1.0, hop_seconds=0.5)
    second = paperlite_features(x, fs=128, frame_seconds=1.0, hop_seconds=0.5)
    assert first.shape == (3, 7, 6, 11)
    assert len(FEATURE_NAMES) == 11
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_correlation_sketch_tracks_shared_signal():
    rng = np.random.default_rng(4)
    independent = rng.normal(size=(2, 8, 128)).astype(np.float32)
    shared = np.repeat(rng.normal(size=(2, 1, 128)), 8, axis=1).astype(np.float32)
    a = paperlite_features(independent, fs=128)[:, 0, 0, -2]
    b = paperlite_features(shared, fs=128)[:, 0, 0, -2]
    assert np.all(b > 0.99)
    assert np.all(b > a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA quantum circuit required")
@pytest.mark.parametrize("classes,channels,expected", [(2, 23, 859), (4, 64, 926), (5, 6, 881), (9, 32, 959)])
def test_model_parameter_budget_and_shape(classes, channels, expected):
    model = PaperLiteQNeuro(classes, channels).cuda()
    # 720 BiLSTM + C channel attention + 2 temporal fusion + 20 LayerNorm
    # + 44 quantum projection + 24 quantum + 13 parameters/KAN output.
    assert count_trainable_parameters(model) == expected
    assert count_trainable_parameters(model) < 1000
    model.eval()
    assert model(torch.randn(3, 7, channels, 11, device="cuda")).shape == (3, classes)
