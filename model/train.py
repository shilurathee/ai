import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents GUI popup errors
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Constants ─────────────────────────────────────────
FEATURE_COLS = [
    'avg_marks', 'courses_enrolled', 'courses_with_zero_marks',
    'assessments_missing', 'grade_f_count', 'lowest_course_score',
    'marks_trend', 'completion_rate'
]
LABEL_COL    = 'dropped_out'
CSV_PATH     = 'data/training_data.csv'
MODEL_PATH   = 'model/rf_model.pkl'
CHART_PATH   = 'model/feature_importance.png'
METRICS_PATH = 'model/metrics.pkl'

# Human-readable labels for the feature importance chart
LABEL_MAP = {
    'avg_marks':               'Average Marks',
    'courses_enrolled':        'Courses Enrolled',
    'courses_with_zero_marks': 'Courses with Zero Marks',
    'assessments_missing':     'Assessments Missing',
    'grade_f_count':           'Failing Courses (F grade)',
    'lowest_course_score':     'Lowest Course Score',
    'marks_trend':             'Marks Trend (slope)',
    'completion_rate':         'Completion Rate',
}


def main():
    # ── STEP 1: Load and validate data ────────────────────
    df = pd.read_csv(CSV_PATH)

    # Validate all required columns exist
    for col in FEATURE_COLS:
        assert col in df.columns, "Missing feature column: " + col
    assert LABEL_COL in df.columns, "Missing label column: " + LABEL_COL
    assert df.shape[0] == 500, "Expected 500 rows, got " + str(df.shape[0])
    assert df[LABEL_COL].isin([0, 1]).all(), "Label column has values other than 0/1"
    assert df[FEATURE_COLS].isnull().sum().sum() == 0, "Feature columns contain NaN values"

    dropout_count = int(df[LABEL_COL].sum())
    dropout_pct = round(df[LABEL_COL].mean() * 100, 1)
    print("✅ Data loaded: 500 rows, 8 features")
    print("📊 Dropout rate: {}% ({} students)".format(dropout_pct, dropout_count))

    # ── STEP 2: Split into train/test ─────────────────────
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("📐 Train size: {} rows".format(len(X_train)))
    print("📐 Test size:  {} rows".format(len(X_test)))

    # ── STEP 3: Train Random Forest ──────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("🌲 Random Forest trained — 100 trees, max_depth=8")

    # ── STEP 4: Evaluate on test set ─────────────────────
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=['Not Dropout', 'Dropout'],
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    # Extract dropout-specific metrics
    precision_dropout = report['Dropout']['precision']
    recall_dropout    = report['Dropout']['recall']
    f1_dropout        = report['Dropout']['f1-score']

    # Print evaluation results
    print("")
    print("══════════════════════════════════")
    print(" MODEL EVALUATION RESULTS")
    print("══════════════════════════════════")
    print("Accuracy  : {:.1f}%".format(accuracy * 100))
    print("Precision : {:.1f}%   (how many flagged-as-dropout are truly at risk)".format(
        precision_dropout * 100))
    print("Recall    : {:.1f}%   (how many actual dropouts were caught)".format(
        recall_dropout * 100))
    print("F1 Score  : {:.1f}%".format(f1_dropout * 100))
    print("")
    print("Confusion Matrix:")
    print("              Predicted")
    print("              No    Yes")
    print("Actual  No  [ {:<4d}  {:<4d}]".format(cm[0][0], cm[0][1]))
    print("        Yes [ {:<4d}  {:<4d}]".format(cm[1][0], cm[1][1]))
    print("══════════════════════════════════")

    # ── STEP 5: Save model ───────────────────────────────
    os.makedirs('model', exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model saved → " + MODEL_PATH)

    # ── STEP 6: Save metrics dict ────────────────────────
    metrics = {
        'accuracy':            round(accuracy * 100, 1),
        'precision':           round(precision_dropout * 100, 1),
        'recall':              round(recall_dropout * 100, 1),
        'f1':                  round(f1_dropout * 100, 1),
        'confusion_matrix':    cm.tolist(),
        'feature_names':       FEATURE_COLS,
        'feature_importances': model.feature_importances_.tolist(),
        'train_size':          len(X_train),
        'test_size':           len(X_test),
        'dropout_rate_pct':    round(y.mean() * 100, 1),
    }

    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f)

    print("✅ Metrics saved → " + METRICS_PATH)

    # ── STEP 7: Plot feature importance chart ────────────
    importances = model.feature_importances_
    indices = np.argsort(importances)
    sorted_features = [FEATURE_COLS[i] for i in indices]
    sorted_values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        [LABEL_MAP[f] for f in sorted_features],
        sorted_values,
        color='#3B82F6',
        edgecolor='white',
        height=0.6,
    )
    ax.set_xlabel('Feature Importance Score', fontsize=11)
    ax.set_title(
        'What Drives Dropout Risk?\nRandom Forest Feature Importances',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlim(0, max(sorted_values) * 1.25)

    # Add value labels on each bar
    for bar, val in zip(bars, sorted_values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            '{:.3f}'.format(val),
            va='center', fontsize=9, color='#374151'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("✅ Chart saved → " + CHART_PATH)

    # ── STEP 8: Final summary ────────────────────────────
    # Get top 3 features by importance
    top3_indices = np.argsort(importances)[::-1][:3]
    print("")
    print("══════════════════════════════════")
    print(" PHASE 3 TRAINING COMPLETE")
    print("══════════════════════════════════")
    print("Model file  : " + MODEL_PATH)
    print("Metrics file: " + METRICS_PATH)
    print("Chart       : " + CHART_PATH)
    print("──────────────────────────────────")
    print("Top 3 most important features:")
    for rank, idx in enumerate(top3_indices, 1):
        print("  {}. {}  →  {:.4f}".format(
            rank, LABEL_MAP[FEATURE_COLS[idx]], importances[idx]))
    print("══════════════════════════════════")


if __name__ == '__main__':
    main()
