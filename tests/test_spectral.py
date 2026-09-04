import numpy as np
from src.features.spectral import spectral_features
def test_spectral_shape_and_finite():
 z=spectral_features(np.random.default_rng(1).normal(size=(2,18,2000)),128);assert z.shape==(2,30,108);assert np.isfinite(z).all()
