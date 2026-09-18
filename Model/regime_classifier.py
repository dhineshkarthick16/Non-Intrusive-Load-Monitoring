"""
regime_classifier.py
--------------------
Regime-Aware Multi-Expert NILM Classifier.
Dynamically routes inference across three specialized expert models based on the
observed NO LOAD baseline floor:
  1. Low Regime (V_baseline in 4.0 - 7.2V):
     Uses Expert 1 trained on 7-minute data (Charger ~ 9.6V, Bulb ~ 10.7V, Both ~ 15.9V).
  2. Mid Regime (V_baseline in 8.0 - 9.8V):
     Uses Expert 2 trained on 40-minute validation + historical data (Charger ~ 12.2V, Bulb ~ 13.7V, Both ~ 18.0V).
  3. High Regime (V_baseline in 10.0 - 12.5V):
     Uses Expert 3 trained on 35-minute data (Charger ~ 13.9V, Bulb ~ 14.1V, Both ~ 19.5V).
  4. Intermediate / Outside Baselines:
     Dynamic Softmax Distance-Weighted Blending of expert predictions using learned
     additive physical invariance (delta_V1 ~ +2.7V Charger, +3.6V Bulb, +9.0V Both).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class RegimeAwareNILMClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-Expert Dynamic Routing NILM Classifier.
    """

    def __init__(self, expert_low=None, expert_mid=None, expert_high=None, v_baseline=6.60):
        self.expert_low = expert_low
        self.expert_mid = expert_mid
        self.expert_high = expert_high
        self.v_baseline = float(v_baseline)
        self.classes_ = np.array(["Both", "Bulb", "Charger", "NO LOAD"])
        self.center_low = 6.60
        self.center_mid = 9.50
        self.center_high = 10.50
        _, self.active_regime, self.active_regime_code = self.get_regime_weights(self.v_baseline)

    def fit(self, X=None, y=None):
        """No-op fit method for scikit-learn estimator interface compliance."""
        return self

    def set_baseline(self, v_base: float):
        """Update current zero-load baseline reference and active regime metadata."""
        self.v_baseline = float(v_base)
        _, self.active_regime, self.active_regime_code = self.get_regime_weights(self.v_baseline)

    def get_regime_weights(self, v_base: float):
        """
        Compute expert routing weights and active regime tag based on baseline floor.
        """
        # Exact regime windows
        if v_base <= 7.5:
            return np.array([1.0, 0.0, 0.0]), "LOW (4-7V) [7-min Model]", "LOW"
        elif 8.0 <= v_base <= 9.8:
            return np.array([0.0, 1.0, 0.0]), "MID (8-9V) [40-min Model]", "MID"
        elif v_base >= 10.2:
            return np.array([0.0, 0.0, 1.0]), "HIGH (10-11V) [35-min Model]", "HIGH"
        else:
            # Intermediate interpolation using Gaussian RBF kernel (sigma=1.2V)
            centers = np.array([self.center_low, self.center_mid, self.center_high])
            diffs = v_base - centers
            weights = np.exp(-(diffs ** 2) / (2.0 * (1.2 ** 2)))
            weights /= np.sum(weights)
            return weights, f"INTERPOLATED (Base={v_base:.1f}V)", "INTERP"

    def _align_expert_probs(self, expert, X_sub):
        """Safely extract and map class probabilities matching self.classes_."""
        p = expert.predict_proba(X_sub)
        if hasattr(expert, "classes_") and np.array_equal(expert.classes_, self.classes_):
            return p
        aligned = np.zeros((len(X_sub), len(self.classes_)))
        exp_classes = getattr(expert, "classes_", self.classes_)
        for i, c in enumerate(exp_classes):
            idx = np.where(self.classes_ == c)[0]
            if len(idx) > 0:
                aligned[:, idx[0]] = p[:, i]
        return aligned

    def predict_proba(self, X):
        """Predict class probabilities for X under current baseline regime."""
        if isinstance(X, pd.DataFrame):
            X_arr = X[["V1", "V2", "V3"]].values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)

        v_base = getattr(self, "v_baseline", 9.50)
        weights, regime_tag, code = self.get_regime_weights(v_base)
        self.active_regime = regime_tag
        self.active_regime_code = code

        probs_total = np.zeros((len(X_arr), len(self.classes_)))

        # 1. Low Expert (7-min data centered at ~6.5V)
        if weights[0] > 0.005 and self.expert_low is not None:
            shift_low = v_base - self.center_low
            X_low = X_arr.copy()
            X_low[:, 0] = np.maximum(0.1, X_low[:, 0] - shift_low)
            p_low = self._align_expert_probs(self.expert_low, X_low)
            probs_total += weights[0] * p_low

        # 2. Mid Expert (40-min + historical centered at ~9.2V)
        if weights[1] > 0.005 and self.expert_mid is not None:
            shift_mid = v_base - self.center_mid
            X_mid = X_arr.copy()
            X_mid[:, 0] = np.maximum(0.1, X_mid[:, 0] - shift_mid)
            p_mid = self._align_expert_probs(self.expert_mid, X_mid)
            probs_total += weights[1] * p_mid

        # 3. High Expert (35-min data centered at ~10.5V)
        if weights[2] > 0.005 and self.expert_high is not None:
            shift_high = v_base - self.center_high
            X_high = X_arr.copy()
            X_high[:, 0] = np.maximum(0.1, X_high[:, 0] - shift_high)
            p_high = self._align_expert_probs(self.expert_high, X_high)
            probs_total += weights[2] * p_high

        # Normalize across classes
        row_sums = probs_total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs_total / row_sums

    def predict(self, X):
        """Predict class labels for X under current baseline regime."""
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]
