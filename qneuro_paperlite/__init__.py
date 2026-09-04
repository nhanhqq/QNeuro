"""Independent PaperLite variant; the existing QNeurov2 code is untouched."""

from .features import FEATURE_NAMES, FEATURE_VERSION, paperlite_features
from .model import PaperLiteQNeuro, count_trainable_parameters

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "paperlite_features",
    "PaperLiteQNeuro",
    "count_trainable_parameters",
]
