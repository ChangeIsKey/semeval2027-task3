"""Confidence Margin Calibration for Subtask 2."""

from typing import List, Tuple
import numpy as np


class ConfidenceMarginCalibrator:
    """
    Applies margin calibration and temperature scaling to raw similarity scores,
    resolving boundary ambiguity near classification decision thresholds.
    """

    def __init__(self, temperature: float = 1.2, margin: float = 0.03):
        self.temperature = temperature
        self.margin = margin

    def calibrate_scores(self, raw_scores: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrates scores and produces hardened binary decisions.
        Returns:
            (calibrated_probs, binary_labels)
        """
        # Center around decision boundary and scale by temperature
        logits = (raw_scores - threshold) / self.temperature
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Default decision is prob >= 0.5 (equivalent to raw_score >= threshold)
        binary_labels = (probs >= 0.5).astype(int)

        return probs, binary_labels
