import numpy as np
import pandas as pd
import random
import os

# Set seeds for full reproducibility
random.seed(42)
np.random.seed(42)


def main():
    n = 500

    # Generate avg_marks — beta distribution centred around 0.6
    avg_marks = np.clip(np.random.beta(5, 2, n) * 0.8 + 0.1, 0.0, 1.0)

    # Generate courses_enrolled — integers 3, 4, or 5
    courses_enrolled = np.random.randint(3, 6, n)

    # Generate courses_with_zero_marks — integers 0 through 3
    courses_with_zero_marks = np.random.randint(0, 4, n)

    # Generate assessments_missing — integers 0 through 8
    assessments_missing = np.random.randint(0, 9, n)

    # Generate grade_f_count — integers 0 through 4
    grade_f_count = np.random.randint(0, 5, n)

    # Generate lowest_course_score — beta skewed low
    lowest_course_score = np.clip(np.random.beta(2, 3, n), 0.0, 1.0)

    # Generate marks_trend — normal distribution centred at 0
    marks_trend = np.round(np.random.normal(0, 0.5, n), 4)

    # Generate completion_rate — beta skewed high
    completion_rate = np.clip(np.random.beta(6, 2, n), 0.0, 1.0)

    # Assign dropout labels using the specified rule
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        if avg_marks[i] < 0.45 and assessments_missing[i] > 3 and completion_rate[i] < 0.60:
            labels[i] = 1

    # Check dropout rate and adjust if needed
    dropout_count = int(labels.sum())
    dropout_rate = dropout_count / n

    if dropout_rate < 0.15:
        # Need to flip some 0s to 1s — pick rows with lowest avg_marks first
        target_count = int(n * 0.20)
        needed = target_count - dropout_count

        # Get indices of label-0 rows, sorted by avg_marks ascending
        zero_indices = np.where(labels == 0)[0]
        sorted_zero = zero_indices[np.argsort(avg_marks[zero_indices])]

        # Flip the 'needed' lowest-avg_marks label-0 rows to 1
        for idx in sorted_zero[:needed]:
            labels[idx] = 1

    elif dropout_rate > 0.30:
        # Need to flip some 1s back to 0 — pick rows with highest avg_marks first
        target_count = int(n * 0.20)
        excess = dropout_count - target_count

        # Get indices of label-1 rows, sorted by avg_marks descending
        one_indices = np.where(labels == 1)[0]
        sorted_ones = one_indices[np.argsort(-avg_marks[one_indices])]

        # Flip the 'excess' highest-avg_marks label-1 rows to 0
        for idx in sorted_ones[:excess]:
            labels[idx] = 0

    # Report the final dropout rate
    final_dropout = int(labels.sum())
    final_rate = final_dropout / n
    print("Dropout rate: {:.1f}% ({}/{})".format(final_rate * 100, final_dropout, n))

    # Build DataFrame with columns in the specified order
    df = pd.DataFrame({
        'avg_marks':               np.round(avg_marks, 4),
        'courses_enrolled':        courses_enrolled.astype(int),
        'courses_with_zero_marks': courses_with_zero_marks.astype(int),
        'assessments_missing':     assessments_missing.astype(int),
        'grade_f_count':           grade_f_count.astype(int),
        'lowest_course_score':     np.round(lowest_course_score, 4),
        'marks_trend':             marks_trend,
        'completion_rate':         np.round(completion_rate, 4),
        'dropped_out':             labels.astype(int),
    })

    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    df.to_csv('data/training_data.csv', index=False)

    # Print confirmation with statistics
    print("")
    print("✅ training_data.csv saved — {} rows, {} columns".format(len(df), len(df.columns)))
    print("📊 Feature summary:")
    print("   avg_marks           — mean: {:.2f},  min: {:.2f},  max: {:.2f}".format(
        df['avg_marks'].mean(), df['avg_marks'].min(), df['avg_marks'].max()))
    print("   assessments_missing — mean: {:.2f},  min: {},     max: {}".format(
        df['assessments_missing'].mean(),
        int(df['assessments_missing'].min()),
        int(df['assessments_missing'].max())))
    print("   completion_rate     — mean: {:.2f},  min: {:.2f},  max: {:.2f}".format(
        df['completion_rate'].mean(), df['completion_rate'].min(), df['completion_rate'].max()))
    print("   dropped_out         — 1s: {},  0s: {}  ({:.1f}% dropout rate)".format(
        int(df['dropped_out'].sum()),
        int((df['dropped_out'] == 0).sum()),
        df['dropped_out'].mean() * 100))


if __name__ == '__main__':
    main()
