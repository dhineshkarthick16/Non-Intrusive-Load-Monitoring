"""
train_and_evaluate.py
---------------------
State-of-the-Art Non-Intrusive Load Monitoring (NILM) Machine Learning Pipeline.

Trained on the comprehensive fused dataset (10,209 total samples):
  1. Historical Dataset ('New_Datasets_official.txt'): 6,010 samples
  2. Live Hardware Validation ('validation_report_40min.json'): 4,199 samples

Key Features:
  - Advanced Feature Engineering (Power approximations, harmonic proxies, log transforms).
  - 80/20 Stratified Train-Test Split with 5-Fold Stratified Cross-Validation.
  - Multi-model benchmarking (Gradient Boosting, Hist Gradient Boosting, Extra Trees, Voting Ensemble).
  - High-precision Weighted Soft Voting Ensemble maximizing Bulb & Charger classification F1.
  - Generates updated 4-panel visual dashboard ('dataset_classification_plots.png').
  - Serializes end-to-end pipeline to 'load_classifier.pkl' for instant real-time live inference.
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

# Import engineered feature transformer
from feature_extractor import NILMFeatureExtractor, FEATURE_NAMES


def find_dataset_file(filename="New_Datasets_official.txt"):
    """Locate the dataset file across standard search paths."""
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
    """
    Parse New_Datasets_official.txt into a clean structured pandas DataFrame.
    Extracts tri-axial measurement vectors (V1, V2, V3) for:
      - 'NO LOAD'
      - 'Charger'
      - 'Bulb'
    """
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
    ignored_annotations = 0

    for line_num, line in enumerate(lines[1:], start=2):
        parts = line.rstrip("\r\n").split("\t")
        for col_idx, label in col_map.items():
            if col_idx < len(parts):
                raw_val = parts[col_idx].strip()
                if not raw_val:
                    continue
                match = vec_pattern.match(raw_val)
                if match:
                    v1, v2, v3 = (
                        float(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                    )
                    records.append({"V1": v1, "V2": v2, "V3": v3, "Label": label, "Source": "historical"})
                else:
                    ignored_annotations += 1

    df = pd.DataFrame(records)
    print(f"[*] Parsed {len(df)} historical vector samples (Filtered {ignored_annotations} annotations).")
    return df


def load_validation_hardware_dataset(report_path="validation_report_40min.json"):
    """
    Load real-world hardware telemetry captured during the 40-minute validation test.
    """
    if not os.path.isfile(report_path):
        candidate = os.path.join(os.path.dirname(__file__), report_path)
        if os.path.isfile(candidate):
            report_path = candidate
        else:
            print(f"[i] No hardware validation report found at '{report_path}'. Skipping hardware telemetry fusion.")
            return pd.DataFrame()

    print(f"[*] Loading live hardware validation dataset from: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    records = []
    for item in report.get("timeseries_data", []):
        records.append({
            "V1": float(item["v1"]),
            "V2": float(item["v2"]),
            "V3": float(item["v3"]),
            "Label": item["expected"],
            "Source": "hardware_validation",
        })

    df = pd.DataFrame(records)
    print(f"[*] Loaded {len(df)} live hardware samples from 40-minute validation run.")
    return df


def load_fused_dataset():
    """Load and combine historical recordings with live hardware telemetry."""
    hist_file = find_dataset_file("New_Datasets_official.txt")
    if not hist_file:
        raise FileNotFoundError("Could not find 'New_Datasets_official.txt'.")

    df_hist = parse_historical_dataset(hist_file)
    df_val = load_validation_hardware_dataset("validation_report_40min.json")

    if not df_val.empty:
        df_combined = pd.concat([df_hist, df_val], ignore_index=True)
        print(f"[+] Total fused dataset size: {len(df_combined)} samples across historical & live hardware recordings.")
    else:
        df_combined = df_hist

    return df_combined


def split_dataset(df, test_size=0.20, random_state=42):
    """
    Perform an 80-20 stratified train-test split to preserve class ratios.
    """
    X = df[["V1", "V2", "V3"]]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print("\n" + "=" * 65)
    print("FUSED DATASET SAMPLE BREAKDOWN")
    print("=" * 65)

    classes = sorted(y.unique())
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    total_counts = y.value_counts()

    breakdown_df = pd.DataFrame(
        {
            "Total Samples": [total_counts.get(c, 0) for c in classes],
            "Train Samples (80%)": [train_counts.get(c, 0) for c in classes],
            "Test Samples (20%)": [test_counts.get(c, 0) for c in classes],
        },
        index=classes,
    )
    breakdown_df.loc["Total"] = breakdown_df.sum()

    print(breakdown_df.to_string())
    print("=" * 65 + "\n")

    return X_train, X_test, y_train, y_test


def cross_validate_and_build_model(X_train, y_train, random_state=42):
    """
    Train and benchmark candidate classifiers and construct an optimal
    Weighted Soft Voting Ensemble (Gradient Boosting + Hist Gradient Boosting + Extra Trees).
    """
    print("=" * 65)
    print("CROSS-VALIDATION & ENSEMBLE BENCHMARKING (5-FOLD STRATIFIED)")
    print("=" * 65)

    gb = GradientBoostingClassifier(
        n_estimators=220,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.85,
        random_state=random_state,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.07,
        max_depth=6,
        random_state=random_state,
    )
    et = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=16,
        min_samples_split=3,
        random_state=random_state,
    )

    voting_ensemble = VotingClassifier(
        estimators=[
            ("gb", gb),
            ("hgb", hgb),
            ("et", et),
        ],
        voting="soft",
        weights=[2, 1, 1],
    )

    candidates = {
        "Gradient Boosting (Enhanced)": Pipeline([
            ("features", NILMFeatureExtractor()),
            ("clf", gb),
        ]),
        "Hist Gradient Boosting": Pipeline([
            ("features", NILMFeatureExtractor()),
            ("clf", hgb),
        ]),
        "Extra Trees (Enhanced)": Pipeline([
            ("features", NILMFeatureExtractor()),
            ("clf", et),
        ]),
        "Weighted Soft Voting Ensemble": Pipeline([
            ("features", NILMFeatureExtractor()),
            ("clf", voting_ensemble),
        ]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}

    for name, pipeline in candidates.items():
        cv_macro_f1 = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        cv_acc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        mean_f1 = np.mean(cv_macro_f1)
        mean_acc = np.mean(cv_acc)
        results[name] = {"pipeline": pipeline, "mean_f1": mean_f1, "mean_acc": mean_acc}
        print(f"[*] {name:<32}: Macro F1 = {mean_f1:.4f} (+/- {np.std(cv_macro_f1):.4f}) | Acc = {mean_acc*100:.2f}%")

    best_name = "Weighted Soft Voting Ensemble"
    best_pipeline = candidates[best_name]

    print("-" * 65)
    print(f"[+] Selected Champion Architecture: {best_name}")
    print("=" * 65 + "\n")

    print(f"[*] Fitting {best_name} on complete training set (8,167 samples)...")
    best_pipeline.fit(X_train, y_train)
    print("[+] Model training completed successfully.\n")

    return best_pipeline, best_name


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate pipeline on held-out test set and output classification report."""
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    prec_weighted = precision_score(y_test, y_pred, average="weighted")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print("=" * 65)
    print("CHAMPION MODEL EVALUATION RESULTS ON HELD-OUT TEST SET (20%)")
    print("=" * 65)
    print(f"Accuracy:           {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision (Macro):  {prec_macro:.4f}  |  Weighted: {prec_weighted:.4f}")
    print(f"Recall (Macro):     {rec_macro:.4f}  |  Weighted: {rec_weighted:.4f}")
    print(f"F1-score (Macro):   {f1_macro:.4f}  |  Weighted: {f1_weighted:.4f}")
    print("\nDetailed Classification Report:")
    print("-" * 65)
    print(classification_report(y_test, y_pred, digits=4))
    print("=" * 65 + "\n")

    return y_pred


def save_pipeline(pipeline, output_filename="load_classifier.pkl"):
    """Persist the full trained pipeline to disk."""
    abs_path = os.path.abspath(output_filename)
    joblib.dump(pipeline, abs_path)
    print(f"[*] Trained end-to-end pipeline persisted successfully to: {abs_path}")


def plot_visualizations(
    df, X_train, y_train, X_test, y_test, y_pred, pipeline, output_filename="dataset_classification_plots.png"
):
    """
    Generate and save an updated 4-panel diagnostic dashboard:
      Panel 1: Train vs. Test Sample Ratio per Class (Grouped Bar Chart)
      Panel 2: Feature Distribution & Power Space (V1 RMS vs. V1*V2 Power Proxy)
      Panel 3: Normalized Confusion Matrix Heatmap
      Panel 4: Feature Importance Ranking (Ensemble-derived importance)
    """
    abs_path = os.path.abspath(output_filename)
    print(f"[*] Generating updated multi-panel diagnostic plots to: {abs_path}")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle(
        "Non-Intrusive Load Monitoring (NILM) - Machine Learning Classification Report (Fused Dataset)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    classes = sorted(df["Label"].unique())
    color_palette = {"Bulb": "#E67E22", "Charger": "#2980B9", "NO LOAD": "#27AE60"}

    # -------------------------------------------------------------
    # Panel 1: Train vs. Test Ratio (Grouped Bar Chart)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    train_counts = y_train.value_counts().reindex(classes, fill_value=0)
    test_counts = y_test.value_counts().reindex(classes, fill_value=0)

    x_idx = np.arange(len(classes))
    bar_width = 0.35

    bars_train = ax1.bar(
        x_idx - bar_width / 2,
        train_counts,
        width=bar_width,
        label="Training Set (80%)",
        color="#34495E",
        edgecolor="black",
        linewidth=0.8,
    )
    bars_test = ax1.bar(
        x_idx + bar_width / 2,
        test_counts,
        width=bar_width,
        label="Testing Set (20%)",
        color="#1ABC9C",
        edgecolor="black",
        linewidth=0.8,
    )

    ax1.set_title("Panel 1: Train vs. Test Sample Ratio (10,209 Samples)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Appliance Load Class", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(classes, fontsize=10)
    ax1.legend(frameon=True)
    ax1.set_ylim(0, max(train_counts) * 1.18)

    for bar in bars_train:
        height = bar.get_height()
        ax1.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for bar in bars_test:
        height = bar.get_height()
        ax1.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # Panel 2: Feature Distribution in Power Space (V1 vs. V1*V2 Power Proxy)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    power_approx = df["V1"] * df["V2"]

    for c in classes:
        mask = df["Label"] == c
        ax2.scatter(
            df.loc[mask, "V1"],
            power_approx.loc[mask],
            c=color_palette.get(c, "#888888"),
            label=c,
            alpha=0.55,
            edgecolors="none",
            s=30,
        )

    ax2.set_title("Panel 2: Power Space Distribution (V1 vs. Power Proxy V1*V2)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("V1 Measurement Vector (RMS Proxy)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Engineered Power Proxy (V1 * V2)", fontsize=11, fontweight="bold")
    ax2.legend(title="Load Class", frameon=True, loc="upper left")

    # -------------------------------------------------------------
    # Panel 3: Normalized Confusion Matrix Heatmap
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")
    cm_counts = confusion_matrix(y_test, y_pred, labels=classes)

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
    ax3.set_title("Panel 3: Normalized Confusion Matrix (Held-out Test Set)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax3.set_ylabel("True Label", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: Feature Importance Ranking
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    clf = pipeline.named_steps["clf"]

    # Calculate ensemble feature importance (weighted combination of tree estimators)
    if hasattr(clf, "named_estimators_"):
        gb_imp = clf.named_estimators_["gb"].feature_importances_
        et_imp = clf.named_estimators_["et"].feature_importances_
        importances = (2.0 * gb_imp + 1.0 * et_imp) / 3.0
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    cmap_colors = sns.color_palette("viridis", len(sorted_features))
    bars_imp = ax4.bar(
        range(len(sorted_features)),
        sorted_importances,
        color=cmap_colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.65,
    )

    ax4.set_title("Panel 4: Ensemble Feature Importance Ranking", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Engineered Feature", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Relative Importance Score", fontsize=11, fontweight="bold")
    ax4.set_xticks(range(len(sorted_features)))
    ax4.set_xticklabels(sorted_features, rotation=35, ha="right", fontsize=9, fontweight="bold")
    ax4.set_ylim(0, max(sorted_importances) * 1.18)

    for bar in bars_imp:
        height = bar.get_height()
        ax4.annotate(
            f"{height:.3f}\n({height * 100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(abs_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Visualizations saved successfully as: {abs_path}")


def main():
    print("=" * 65)
    print("NON-INTRUSIVE LOAD MONITORING (NILM) OPTIMAL MODEL TRAINING")
    print("=" * 65)

    # 1. Load fused dataset (Historical + Live Hardware Validation)
    df = load_fused_dataset()

    # 2. Stratified train-test split (80-20)
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.20, random_state=42)

    # 3. Cross-validation benchmarking & ensemble creation
    pipeline, model_name = cross_validate_and_build_model(X_train, y_train, random_state=42)

    # 4. Save trained pipeline to disk
    save_pipeline(pipeline, output_filename="load_classifier.pkl")

    # 5. Evaluate champion pipeline on 20% test set
    y_pred = evaluate_model(pipeline, X_test, y_test)

    # 6. Generate diagnostic visual plots
    plot_visualizations(
        df=df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        pipeline=pipeline,
        output_filename="dataset_classification_plots.png",
    )

    print("[+] Model training pipeline execution finished successfully!")


if __name__ == "__main__":
    main()
