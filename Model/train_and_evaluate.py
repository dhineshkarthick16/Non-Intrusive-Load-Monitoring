"""
train_and_evaluate.py
---------------------
State-of-the-Art Non-Intrusive Load Monitoring (NILM) Multi-Expert Training Pipeline.

Implements the Regime-Aware Multi-Expert Architecture:
  - Rule 1 (Low Regime, NO LOAD ~ 4-6V / 7V): Expert 1 trained on 7-minute 4-class dataset.
  - Rule 2 (Mid Regime, NO LOAD ~ 8-9V): Expert 2 trained on 40-minute validation + historical dataset.
  - Rule 3 (High Regime, NO LOAD ~ 10-11V): Expert 3 trained on 35-minute 4-class dataset.
  - Rule 4 (Intermediate / Drift Baselines): Dynamic soft Gaussian RBF blending and physics-invariant shift.

Serializes the complete RegimeAwareNILMClassifier to 'load_classifier.pkl'.
Generates updated 4-panel diagnostic dashboard ('dataset_classification_plots.png').
"""

import json
import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Import engineered feature transformer and multi-expert classifier
from feature_extractor import NILMFeatureExtractor, FEATURE_NAMES
from regime_classifier import RegimeAwareNILMClassifier


def find_dataset_file(filename):
    """Locate dataset file across standard search paths."""
    candidate_paths = [
        filename,
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join("..", "Firmware", filename),
        os.path.join(os.path.dirname(__file__), "..", "Firmware", filename),
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def parse_historical_dataset(filepath):
    """Parse New_Datasets_official.txt into structured DataFrame."""
    print(f"[*] Loading historical dataset from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"File {filepath} is empty.")

    vec_pattern = re.compile(
        r'^\s*"?\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*"?\s*$'
    )

    header_line = lines[0].rstrip("\r\n")
    header_tokens = [tok.strip() for tok in header_line.split("\t")]

    col_map = {}
    for idx, tok in enumerate(header_tokens):
        tok_upper = tok.upper()
        if "NO LOAD" in tok_upper:
            col_map[idx] = "NO LOAD"
        elif "CHARGER" in tok_upper:
            col_map[idx] = "Charger"
        elif "BULB" in tok_upper:
            col_map[idx] = "Bulb"

    if len(col_map) < 3:
        col_map = {0: "NO LOAD", 3: "Charger", 6: "Bulb"}

    records = []
    ignored = 0
    for line in lines[1:]:
        parts = line.rstrip("\r\n").split("\t")
        for col_idx, label in col_map.items():
            if col_idx < len(parts):
                raw_val = parts[col_idx].strip()
                if not raw_val:
                    continue
                m = vec_pattern.match(raw_val)
                if m:
                    v1, v2, v3 = float(m.group(1)), float(m.group(2)), float(m.group(3))
                    records.append({"V1": v1, "V2": v2, "V3": v3, "Label": label})
                else:
                    ignored += 1

    df = pd.DataFrame(records)
    print(f"[*] Parsed {len(df)} historical vector samples (Filtered {ignored} annotations).")
    return df


def load_validation_hardware_dataset(report_path="validation_report_40min.json"):
    """Load real-world hardware telemetry from 40-minute validation report."""
    cand = find_dataset_file(report_path)
    if not cand:
        print(f"[i] Hardware validation report not found at '{report_path}'. Skipping.")
        return pd.DataFrame()

    print(f"[*] Loading 40-minute validation dataset from: {cand}")
    with open(cand, "r", encoding="utf-8") as f:
        report = json.load(f)

    records = []
    for item in report.get("timeseries_data", []):
        records.append({
            "V1": float(item["v1"]),
            "V2": float(item["v2"]),
            "V3": float(item["v3"]),
            "Label": item["expected"],
        })

    df = pd.DataFrame(records)
    print(f"[*] Loaded {len(df)} live hardware samples from 40-minute validation run.")
    return df


def load_low_regime_dataset(filepath="dataset_7min_4class.csv"):
    """
    Load Low Regime Dataset (~4-6V / 7V):
    Collected during the 7-minute 7-phase hardware collection routine.
    """
    cand = find_dataset_file(filepath)
    if not cand:
        raise FileNotFoundError(f"Could not find '{filepath}'.")
    df = pd.read_csv(cand)
    print(f"[*] Low regime dataset loaded from {cand}: {len(df)} samples across {sorted(df['Label'].unique())}")
    return df[["V1", "V2", "V3", "Label"]]


def load_high_regime_dataset(filepath="dataset_35min_4class.csv"):
    """
    Load High Regime Dataset (~10-11V):
    Collected during the 35-minute hardware collection routine.
    """
    cand = find_dataset_file(filepath)
    if not cand:
        raise FileNotFoundError(f"Could not find '{filepath}'.")
    df = pd.read_csv(cand)
    print(f"[*] High regime dataset loaded from {cand}: {len(df)} samples across {sorted(df['Label'].unique())}")
    return df[["V1", "V2", "V3", "Label"]]


def load_mid_regime_dataset():
    """
    Load Mid Regime Dataset (~8-9.5V):
    Combines 40-minute live validation report + historical recordings,
    augmented with dual-load ('Both') samples using physical delta invariance.
    """
    hist_file = find_dataset_file("New_Datasets_official.txt")
    df_hist = parse_historical_dataset(hist_file) if hist_file else pd.DataFrame()
    df_val = load_validation_hardware_dataset("validation_report_40min.json")

    records = []
    if not df_val.empty:
        for _, row in df_val.iterrows():
            records.append({"V1": float(row["V1"]), "V2": float(row["V2"]), "V3": float(row["V3"]), "Label": row["Label"]})
    if not df_hist.empty:
        for _, row in df_hist.iterrows():
            records.append({"V1": float(row["V1"]), "V2": float(row["V2"]), "V3": float(row["V3"]), "Label": row["Label"]})

    df_mid = pd.DataFrame(records)

    # Mid regime dual-load augmentation ('Both') using physical delta (~ +9.0V V1, V3 ~ 3.8)
    nl_mid = df_mid[df_mid["Label"] == "NO LOAD"]
    if len(nl_mid) > 0:
        nl_samples = nl_mid.sample(n=min(600, len(nl_mid)), random_state=42)
        both_records = []
        for _, row in nl_samples.iterrows():
            v1_both = row["V1"] + 9.0 + np.random.normal(0, 0.4)
            v2_both = v1_both * (3.8 + np.random.normal(0, 0.2))
            v3_both = v2_both / v1_both
            both_records.append({"V1": v1_both, "V2": v2_both, "V3": v3_both, "Label": "Both"})
        df_both = pd.DataFrame(both_records)
        df_mid = pd.concat([df_mid, df_both], ignore_index=True)

    print(f"[*] Mid regime dataset prepared: {len(df_mid)} samples across {sorted(df_mid['Label'].unique())}")
    return df_mid


def build_expert_pipeline(v_base, n_est=180, random_state=42):
    """Construct an expert classification pipeline tuned for a specific baseline center."""
    gb = GradientBoostingClassifier(
        n_estimators=n_est,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=random_state,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=n_est,
        learning_rate=0.08,
        max_depth=6,
        random_state=random_state,
    )
    et = ExtraTreesClassifier(
        n_estimators=n_est,
        max_depth=16,
        min_samples_split=3,
        random_state=random_state,
        n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("gb", gb), ("et", et), ("hgb", hgb)],
        voting="soft",
        weights=[2, 2, 1],
    )
    return Pipeline([
        ("features", NILMFeatureExtractor(v_baseline=v_base)),
        ("clf", ensemble),
    ])


def save_pipeline(pipeline, output_filename="load_classifier.pkl"):
    """Persist the full trained multi-expert pipeline to disk."""
    abs_path = os.path.abspath(output_filename)
    joblib.dump(pipeline, abs_path)
    print(f"[*] Trained multi-expert pipeline persisted successfully to: {abs_path}")


def plot_regime_visualizations(
    master_model,
    df_low, df_mid, df_high,
    y_test_combined, y_pred_combined,
    metrics_summary,
    output_filename="dataset_classification_plots.png"
):
    """
    Generate and save a 4-panel diagnostic dashboard reflecting the multi-expert NILM architecture:
      Panel 1: Dynamic Multi-Expert Routing Weights across Baseline Floors (4V - 12V).
      Panel 2: Physical Load Invariance Across Regimes (Delta V1 vs. Appliance Class).
      Panel 3: Master Multi-Expert Normalized Confusion Matrix (Combined held-out test sets).
      Panel 4: Per-Rule Performance Benchmark (Accuracy & Macro F1 for Rule 1, 2, 3, Combined).
    """
    abs_path = os.path.abspath(output_filename)
    print(f"[*] Generating updated multi-panel diagnostic plots to: {abs_path}")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle(
        "EdgeWatt NILM - Regime-Aware Multi-Expert Classifier Performance Report",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # -------------------------------------------------------------
    # Panel 1: Multi-Regime Routing Weights vs. Baseline Floor
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    base_range = np.linspace(4.0, 12.0, 300)
    w_low, w_mid, w_high = [], [], []

    for vb in base_range:
        w, _, _ = master_model.get_regime_weights(vb)
        w_low.append(w[0])
        w_mid.append(w[1])
        w_high.append(w[2])

    ax1.plot(base_range, w_low, label="Low Expert (7-min: 4-6V / 7V)", color="#27AE60", linewidth=2.8)
    ax1.plot(base_range, w_mid, label="Mid Expert (40-min: 8-9V)", color="#2980B9", linewidth=2.8)
    ax1.plot(base_range, w_high, label="High Expert (35-min: 10-11V)", color="#8E44AD", linewidth=2.8)

    ax1.axvspan(4.0, 7.3, color="#27AE60", alpha=0.10, label="Low Regime Zone")
    ax1.axvspan(8.0, 9.8, color="#2980B9", alpha=0.10, label="Mid Regime Zone")
    ax1.axvspan(10.2, 12.0, color="#8E44AD", alpha=0.10, label="High Regime Zone")

    ax1.set_title("Panel 1: Dynamic Multi-Expert Routing & Blending Weights", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Zero-Load Baseline Voltage (V_baseline)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Routing Probability / Weight", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(4.0, 12.0)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Physical Load Invariance Across Regimes (Delta V1 Boxplots)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    # Compute Delta V1 relative to each regime's baseline
    delta_records = []
    for _, r in df_low.iterrows():
        delta_records.append({"Class": r["Label"], "Delta V1 (V)": r["V1"] - 7.02, "Regime": "Low (7-min)"})
    for _, r in df_mid.iterrows():
        delta_records.append({"Class": r["Label"], "Delta V1 (V)": r["V1"] - 10.15, "Regime": "Mid (40-min)"})
    for _, r in df_high.iterrows():
        delta_records.append({"Class": r["Label"], "Delta V1 (V)": r["V1"] - 10.49, "Regime": "High (35-min)"})

    df_delta = pd.DataFrame(delta_records)
    class_order = ["NO LOAD", "Charger", "Bulb", "Both"]

    sns.boxplot(
        data=df_delta,
        x="Class",
        y="Delta V1 (V)",
        hue="Regime",
        order=class_order,
        palette={"Low (7-min)": "#27AE60", "Mid (40-min)": "#2980B9", "High (35-min)": "#8E44AD"},
        ax=ax2,
        boxprops=dict(alpha=0.85),
    )
    ax2.set_title("Panel 2: Physical Load Delta Invariance Across Regimes", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Appliance Load Class", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Delta V1 = V1 - V_baseline (V)", fontsize=11, fontweight="bold")
    ax2.axhline(0.0, color="#7F8C8D", linestyle="--", linewidth=1.0)
    ax2.legend(title="Dataset Regime", loc="upper left", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Master Multi-Expert Normalized Confusion Matrix
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    classes = class_order
    cm = confusion_matrix(y_test_combined, y_pred_combined, labels=classes, normalize="true")
    cm_counts = confusion_matrix(y_test_combined, y_pred_combined, labels=classes)

    annot_labels = [
        [f"{cm[i, j]:.1%}\n({cm_counts[i, j]})" for j in range(len(classes))]
        for i in range(len(classes))
    ]

    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax3,
        linewidths=1.0,
        linecolor="white",
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    ax3.set_title("Panel 3: Normalized Confusion Matrix (Combined Multi-Expert Test Sets)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Predicted Appliance Label", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Ground-Truth Appliance Label", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: Per-Rule Benchmark Metrics (Accuracy & Macro F1)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    rule_names = list(metrics_summary.keys())
    acc_vals = [metrics_summary[r]["Accuracy"] * 100 for r in rule_names]
    f1_vals = [metrics_summary[r]["Macro_F1"] * 100 for r in rule_names]

    x_idx = np.arange(len(rule_names))
    bar_width = 0.35

    bars_acc = ax4.bar(
        x_idx - bar_width / 2,
        acc_vals,
        width=bar_width,
        label="Test Accuracy (%)",
        color="#2980B9",
        edgecolor="black",
        linewidth=0.8,
    )
    bars_f1 = ax4.bar(
        x_idx + bar_width / 2,
        f1_vals,
        width=bar_width,
        label="Macro F1 Score (%)",
        color="#27AE60",
        edgecolor="black",
        linewidth=0.8,
    )

    ax4.set_title("Panel 4: Per-Regime & Overall Performance Benchmarks", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Operational Rule / Regime", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(rule_names, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax4.set_ylim(85, 103)
    ax4.legend(loc="lower right", frameon=True)

    for bar in bars_acc:
        h = bar.get_height()
        ax4.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    for bar in bars_f1:
        h = bar.get_height()
        ax4.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(abs_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Visualizations saved successfully as: {abs_path}")


def main():
    print("=" * 75)
    print("NON-INTRUSIVE LOAD MONITORING (NILM) REGIME-AWARE MULTI-EXPERT TRAINING")
    print("=" * 75)

    # 1. Load the three distinct regime datasets
    print("\n[Step 1/5] Loading Regime Datasets...")
    df_low = load_low_regime_dataset("dataset_7min_4class.csv")
    df_mid = load_mid_regime_dataset()
    df_high = load_high_regime_dataset("dataset_35min_4class.csv")

    print(f"  - Low Regime (7-min) Samples:   {len(df_low):,}")
    print(f"  - Mid Regime (40-min) Samples:  {len(df_mid):,}")
    print(f"  - High Regime (35-min) Samples: {len(df_high):,}")
    total_samples = len(df_low) + len(df_mid) + len(df_high)
    print(f"  - Total Corpus:                 {total_samples:,} samples")

    # 2. Stratified 80/20 train-test splits for each regime
    print("\n[Step 2/5] Creating Stratified 80/20 Train-Test Splits per Regime...")
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(
        df_low[["V1", "V2", "V3"]], df_low["Label"], test_size=0.20, stratify=df_low["Label"], random_state=42
    )
    X_train_mid, X_test_mid, y_train_mid, y_test_mid = train_test_split(
        df_mid[["V1", "V2", "V3"]], df_mid["Label"], test_size=0.20, stratify=df_mid["Label"], random_state=42
    )
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(
        df_high[["V1", "V2", "V3"]], df_high["Label"], test_size=0.20, stratify=df_high["Label"], random_state=42
    )

    # 3. Train the three expert pipelines
    print("\n[Step 3/5] Training Specialized Expert Pipelines...")
    print("  [*] [1/3] Training Expert 1 (Low Regime, Center=6.60V) on 7-min dataset...")
    expert_low = build_expert_pipeline(v_base=6.60, n_est=180, random_state=42)
    expert_low.fit(X_train_low, y_train_low)

    print("  [*] [2/3] Training Expert 2 (Mid Regime, Center=9.50V) on 40-min/historical dataset...")
    expert_mid = build_expert_pipeline(v_base=9.50, n_est=180, random_state=42)
    expert_mid.fit(X_train_mid, y_train_mid)

    print("  [*] [3/3] Training Expert 3 (High Regime, Center=10.50V) on 35-min dataset...")
    expert_high = build_expert_pipeline(v_base=10.50, n_est=180, random_state=42)
    expert_high.fit(X_train_high, y_train_high)

    # 4. Construct Master Multi-Expert Classifier
    print("\n[Step 4/5] Assembling Master RegimeAwareNILMClassifier...")
    master = RegimeAwareNILMClassifier(
        expert_low=expert_low,
        expert_mid=expert_mid,
        expert_high=expert_high,
        v_baseline=6.60,
    )

    # 5. Evaluate all 4 rules
    print("\n" + "=" * 75)
    print("EVALUATING SYSTEM ACROSS ALL 4 PHYSICAL OPERATIONAL RULES")
    print("=" * 75)

    metrics_summary = {}

    # Rule 1: Low Regime (Base ~ 6.6V)
    master.set_baseline(6.60)
    preds_low = master.predict(X_test_low)
    acc_low = accuracy_score(y_test_low, preds_low)
    f1_low = f1_score(y_test_low, preds_low, average="macro")
    metrics_summary["Rule 1 (Low)"] = {"Accuracy": acc_low, "Macro_F1": f1_low}
    print(f"\n[RULE 1 - Low Regime (NO LOAD ~ 4-6V / 6.6V)]")
    print(f"  Active Tag: {master.active_regime}")
    print(f"  Accuracy:   {acc_low * 100:.2f}% | Macro F1: {f1_low * 100:.2f}%")
    print(classification_report(y_test_low, preds_low, digits=4))

    # Rule 2: Mid Regime (Base ~ 9.5V)
    master.set_baseline(9.50)
    preds_mid = master.predict(X_test_mid)
    acc_mid = accuracy_score(y_test_mid, preds_mid)
    f1_mid = f1_score(y_test_mid, preds_mid, average="macro")
    metrics_summary["Rule 2 (Mid)"] = {"Accuracy": acc_mid, "Macro_F1": f1_mid}
    print(f"\n[RULE 2 - Mid Regime (NO LOAD ~ 8-9V)]")
    print(f"  Active Tag: {master.active_regime}")
    print(f"  Accuracy:   {acc_mid * 100:.2f}% | Macro F1: {f1_mid * 100:.2f}%")
    print(classification_report(y_test_mid, preds_mid, digits=4))

    # Rule 3: High Regime (Base ~ 10.5V)
    master.set_baseline(10.50)
    preds_high = master.predict(X_test_high)
    acc_high = accuracy_score(y_test_high, preds_high)
    f1_high = f1_score(y_test_high, preds_high, average="macro")
    metrics_summary["Rule 3 (High)"] = {"Accuracy": acc_high, "Macro_F1": f1_high}
    print(f"\n[RULE 3 - High Regime (NO LOAD ~ 10-11V)]")
    print(f"  Active Tag: {master.active_regime}")
    print(f"  Accuracy:   {acc_high * 100:.2f}% | Macro F1: {f1_high * 100:.2f}%")
    print(classification_report(y_test_high, preds_high, digits=4))

    # Rule 4: Intermediate Baseline Test (e.g. 7.60V)
    master.set_baseline(7.60)
    sample_tests = [
        {"name": "NO LOAD (Base=7.6V)", "sample": [7.60, 42.0, 5.5], "expected": "NO LOAD"},
        {"name": "Charger (+2.6V on 7.6V)", "sample": [10.20, 49.0, 4.8], "expected": "Charger"},
        {"name": "Bulb (+3.6V on 7.6V)", "sample": [11.20, 58.0, 5.1], "expected": "Bulb"},
        {"name": "Both (+9.0V on 7.6V)", "sample": [16.60, 62.0, 3.8], "expected": "Both"},
    ]
    print(f"\n[RULE 4 - Intermediate Baseline (Base = 7.60V)]")
    print(f"  Active Tag: {master.active_regime}")
    rule4_correct = 0
    for stest in sample_tests:
        p = master.predict([stest["sample"]])[0]
        prob = float(np.max(master.predict_proba([stest["sample"]])) * 100)
        is_ok = p == stest["expected"]
        if is_ok:
            rule4_correct += 1
        print(f"  Sample {stest['name']:<25}: Predicted = {p:<10} ({prob:>5.1f}%) | [{'PASS' if is_ok else 'FAIL'}]")

    # Combined metrics across test sets
    y_test_comb = pd.concat([y_test_low, y_test_mid, y_test_high], ignore_index=True)
    y_pred_comb = np.concatenate([preds_low, preds_mid, preds_high])
    acc_comb = accuracy_score(y_test_comb, y_pred_comb)
    f1_comb = f1_score(y_test_comb, y_pred_comb, average="macro")
    metrics_summary["Overall Combined"] = {"Accuracy": acc_comb, "Macro_F1": f1_comb}

    print("\n" + "=" * 75)
    print(f"FINAL COMBINED MULTI-EXPERT ACCURACY: {acc_comb * 100:.2f}% | MACRO F1: {f1_comb * 100:.2f}%")
    print("=" * 75)

    # 6. Save model to load_classifier.pkl (reset baseline to nominal 6.60V for default startup)
    master.set_baseline(6.60)
    save_pipeline(master, output_filename="load_classifier.pkl")

    # 7. Generate diagnostic plots
    plot_regime_visualizations(
        master_model=master,
        df_low=df_low,
        df_mid=df_mid,
        df_high=df_high,
        y_test_combined=y_test_comb,
        y_pred_combined=y_pred_comb,
        metrics_summary=metrics_summary,
        output_filename="dataset_classification_plots.png",
    )

    print("\n[+] Regime-Aware Multi-Expert NILM training and serialization completed successfully!")


if __name__ == "__main__":
    main()
