"""
feature_extractor.py
--------------------
Defines NILMFeatureExtractor for scikit-learn Pipeline compatibility.
Engineers non-linear power approximations, feature ratios, and logarithmic 
transforms from raw tri-axial (V1: RMS, V2: Peak, V3: Crest Factor) sensor data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    "V1",
    "V2",
    "V3",
    "V1_div_V2",
    "V1_mul_V3",
    "V1_mul_V2",
    "log_power",
    "log_V1",
    "log_V2",
    "V2_div_V1",
    "V1_sub_baseline",
    "power_above_baseline",
    "log_power_above_baseline",
]


class NILMFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transforms raw 3-dimensional load vectors (V1, V2, V3) into 13 engineered features:
      1. V1: RMS voltage/current proxy
      2. V2: Peak amplitude
      3. V3: Crest factor (Peak / RMS)
      4. V1_div_V2: Inverted crest factor ratio (V1 / (V2 + 1e-5))
      5. V1_mul_V3: RMS x Crest factor
      6. V1_mul_V2: Apparent energy / power proxy (RMS x Peak)
      7. log_power: Logarithmic power transformation log1p(V1 * V2)
      8. log_V1: Logarithmic RMS transformation log1p(V1)
      9. log_V2: Logarithmic Peak transformation log1p(V2)
      10. V2_div_V1: Direct Peak to RMS ratio
      11. V1_sub_baseline: V1 - V_baseline (Dynamic zero-drift mitigation)
      12. power_above_baseline: max(0, V1 - V_baseline) * V2
      13. log_power_above_baseline: log1p(max(0, V1 - V_baseline) * V2)
    """

    def __init__(self, v_baseline=9.50):
        self.v_baseline = float(v_baseline)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Check for column names
            if "V1" in X.columns and "V2" in X.columns and "V3" in X.columns:
                v1 = X["V1"].values.astype(float)
                v2 = X["V2"].values.astype(float)
                v3 = X["V3"].values.astype(float)
            else:
                arr = X.values.astype(float)
                v1, v2, v3 = arr[:, 0], arr[:, 1], arr[:, 2]
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                if len(arr) != 3:
                    raise ValueError(f"Expected 3 feature inputs, got {len(arr)}")
                v1 = np.array([arr[0]])
                v2 = np.array([arr[1]])
                v3 = np.array([arr[2]])
            else:
                v1 = arr[:, 0]
                v2 = arr[:, 1]
                v3 = arr[:, 2]

        # 1. Feature ratios
        v1_div_v2 = v1 / (v2 + 1e-5)
        v2_div_v1 = v2 / (v1 + 1e-5)
        v1_mul_v3 = v1 * v3

        # 2. Power approximations
        v1_mul_v2 = v1 * v2

        # 3. Non-linear & logarithmic transformations
        log_power = np.log1p(np.maximum(0.0, v1_mul_v2))
        log_v1 = np.log1p(np.maximum(0.0, v1))
        log_v2 = np.log1p(np.maximum(0.0, v2))

        # 4. Baseline drift mitigation features
        v_base = getattr(self, "v_baseline", 9.2)
        v1_sub_baseline = v1 - v_base
        v1_excess = np.maximum(0.0, v1_sub_baseline)
        power_above_baseline = v1_excess * v2
        log_power_above_baseline = np.log1p(power_above_baseline)

        engineered = np.column_stack([
            v1,
            v2,
            v3,
            v1_div_v2,
            v1_mul_v3,
            v1_mul_v2,
            log_power,
            log_v1,
            log_v2,
            v2_div_v1,
            v1_sub_baseline,
            power_above_baseline,
            log_power_above_baseline,
        ])

        return engineered

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
