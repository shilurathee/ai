"""
export_stats.py
---------------
Queries the database and metrics.pkl to print a clean summary of all
numbers needed to fill into the report and PPT placeholders.

Run from inside dropout_predictor/:
    python report/export_stats.py
"""

import os
import sys
import sqlite3
import pickle

# Make model.predict importable from any working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.predict import predict_all_students, FEATURE_COLS

# Paths
DB_PATH      = 'dropout_predictor.db'
METRICS_PATH = 'model/metrics.pkl'


def main():
    # DATABASE STATS
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total_students    = cursor.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    total_courses     = cursor.execute("SELECT COUNT(*) FROM courses").fetchone()[0]
    total_enrollments = cursor.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]
    total_assessments = cursor.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    null_assessments  = cursor.execute(
        "SELECT COUNT(*) FROM assessments WHERE obtained_marks IS NULL"
    ).fetchone()[0]

    null_rate = round(null_assessments / total_assessments * 100, 1) if total_assessments > 0 else 0.0

    # MODEL PERFORMANCE
    with open(METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)

    accuracy  = metrics['accuracy']
    precision = metrics['precision']
    recall    = metrics['recall']
    f1        = metrics['f1']
    train_size = metrics.get('train_size', 400)
    test_size  = metrics.get('test_size', 100)
    dropout_rate_pct = metrics.get('dropout_rate_pct', '~20')

    # STUDENT RISK DISTRIBUTION
    all_results = predict_all_students(conn)

    high_count   = sum(1 for r in all_results if r['risk_category'] == 'High')
    medium_count = sum(1 for r in all_results if r['risk_category'] == 'Medium')
    low_count    = sum(1 for r in all_results if r['risk_category'] == 'Low')
    total = len(all_results)

    high_pct   = round(high_count / total * 100, 1) if total > 0 else 0
    medium_pct = round(medium_count / total * 100, 1) if total > 0 else 0
    low_pct    = round(low_count / total * 100, 1) if total > 0 else 0

    # TOP 3 FEATURES BY IMPORTANCE
    feature_names  = metrics.get('feature_names', FEATURE_COLS)
    importances    = metrics.get('feature_importances', [0] * 8)
    sorted_indices = sorted(range(len(importances)),
                            key=lambda i: importances[i], reverse=True)

    # PRINT REPORT
    print("")
    print("==================================================")
    print("STATS FOR REPORT -- STUDENT DROPOUT PREDICTOR")
    print("==================================================")
    print("")

    print("-- DATABASE STATS --------------------------------")
    print("Total students:           {}".format(total_students))
    print("Total courses:            {}".format(total_courses))
    print("Total enrollments:        {}".format(total_enrollments))
    print("Total assessments:        {}".format(total_assessments))
    print("Assessments with NULL:    {}".format(null_assessments))
    print("NULL rate:                {}%".format(null_rate))
    print("")

    print("-- MODEL PERFORMANCE -----------------------------")
    print("Accuracy:                 {}%".format(accuracy))
    print("Precision (dropout):      {}%".format(precision))
    print("Recall (dropout):         {}%".format(recall))
    print("F1 Score (dropout):       {}%".format(f1))
    print("Training rows:            {}".format(train_size))
    print("Test rows:                {}".format(test_size))
    print("Dropout rate in CSV:      ~{}%".format(dropout_rate_pct))
    print("")

    print("-- STUDENT RISK DISTRIBUTION ---------------------")
    print("High risk students:       {}  ({}%)".format(high_count, high_pct))
    print("Medium risk students:     {}  ({}%)".format(medium_count, medium_pct))
    print("Low risk students:        {}  ({}%)".format(low_count, low_pct))
    print("")

    print("-- TOP 3 FEATURES BY IMPORTANCE ------------------")
    for rank, idx in enumerate(sorted_indices[:3], 1):
        print("{}. {:<28s} importance: {:.4f}".format(
            rank, feature_names[idx], importances[idx]))
    print("")

    print("-- FILL THESE INTO YOUR REPORT -------------------")
    print("Replace [ACCURACY]  with: {}%".format(accuracy))
    print("Replace [PRECISION] with: {}%".format(precision))
    print("Replace [RECALL]    with: {}%".format(recall))
    print("Replace [F1]        with: {}%".format(f1))
    print("Replace [H]         with: {}".format(high_count))
    print("Replace [H%]        with: {}".format(high_pct))
    print("Replace [M]         with: {}".format(medium_count))
    print("Replace [M%]        with: {}".format(medium_pct))
    print("Replace [L]         with: {}".format(low_count))
    print("Replace [L%]        with: {}".format(low_pct))
    print("Replace [ACC]       with: {}".format(accuracy))
    print("Replace [PREC]      with: {}".format(precision))
    print("Replace [REC]       with: {}".format(recall))
    print("==================================================")

    conn.close()


if __name__ == '__main__':
    main()
