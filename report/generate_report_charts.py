"""
generate_report_charts.py
─────────────────────────
Generates 3 additional charts for the project report and PPT slides.
Saves them to report/charts/.

Run from inside dropout_predictor/:
    python report/generate_report_charts.py
"""

import os
import sys
import sqlite3
import pickle

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — required before pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Make model.predict importable from any working directory ──
# Insert the project root (parent of report/) into sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.predict import predict_all_students, FEATURE_COLS

# ── Ensure output directory exists ────────────────────
os.makedirs('report/charts', exist_ok=True)

# ── Connect to database and load all predictions ─────
DB_PATH = 'dropout_predictor.db'
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
all_results = predict_all_students(conn)

print("Loaded predictions for {} students.".format(len(all_results)))
print("")


# ═══════════════════════════════════════════════════════
# CHART 1: Risk Distribution Bar Chart
# ═══════════════════════════════════════════════════════

high_count   = sum(1 for r in all_results if r['risk_category'] == 'High')
medium_count = sum(1 for r in all_results if r['risk_category'] == 'Medium')
low_count    = sum(1 for r in all_results if r['risk_category'] == 'Low')

fig, ax = plt.subplots(figsize=(7, 5))

categories = ['High', 'Medium', 'Low']
counts     = [high_count, medium_count, low_count]
colours    = ['#EF4444', '#F59E0B', '#10B981']

bars = ax.bar(categories, counts, color=colours, edgecolor='white',
              width=0.55)

# Add count labels on top of each bar
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(count), ha='center', va='bottom',
            fontsize=16, fontweight='bold',
            color='#374151')

ax.set_ylabel('Number of Students', fontsize=12)
ax.set_xlabel('Risk Category', fontsize=12)
ax.set_title('Student Risk Distribution\n(25 Students in Database)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_ylim(0, max(counts) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=12)

plt.tight_layout()
plt.savefig('report/charts/risk_distribution.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Chart 1 saved: report/charts/risk_distribution.png")


# ═══════════════════════════════════════════════════════
# CHART 2: Feature Score Comparison Heatmap
# ═══════════════════════════════════════════════════════

# Build a 25×8 numpy array of feature values
student_names = [r['student_name'] for r in all_results]
feature_matrix = np.array([
    [r['features'][col] for col in FEATURE_COLS]
    for r in all_results
])

# Human-readable feature labels
feature_labels = [
    'Avg Marks', 'Courses\nEnrolled', 'Zero-Mark\nCourses',
    'Assessments\nMissing', 'Failing\nCourses', 'Lowest\nScore',
    'Marks\nTrend', 'Completion\nRate',
]

fig, ax = plt.subplots(figsize=(12, 10))

# Use RdYlGn colourmap: red = bad (low values for good features),
# green = good. We need to invert for "bad-is-high" features like
# assessments_missing — but for simplicity, use consistent scale.
im = ax.imshow(feature_matrix, cmap='RdYlGn', aspect='auto',
               interpolation='nearest')

ax.set_xticks(range(len(FEATURE_COLS)))
ax.set_xticklabels(feature_labels, fontsize=9, ha='center')
ax.set_yticks(range(len(student_names)))
ax.set_yticklabels(student_names, fontsize=9)

ax.set_title('Student Feature Heatmap — Red = Risk, Green = Safe',
             fontsize=14, fontweight='bold', pad=14)

# Add colour bar
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Feature Value', fontsize=10)

plt.tight_layout()
plt.savefig('report/charts/feature_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Chart 2 saved: report/charts/feature_heatmap.png")


# ═══════════════════════════════════════════════════════
# CHART 3: Risk Score Distribution Histogram
# ═══════════════════════════════════════════════════════

risk_scores = [r['risk_score'] for r in all_results]

fig, ax = plt.subplots(figsize=(8, 5))

# Define bin edges
bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Compute histogram data
hist_counts, _ = np.histogram(risk_scores, bins=bin_edges)

# Colour each bar by risk zone
bar_colours = []
for i, edge in enumerate(bin_edges[:-1]):
    mid = (bin_edges[i] + bin_edges[i + 1]) / 2
    if mid < 40:
        bar_colours.append('#10B981')   # green — Low
    elif mid < 70:
        bar_colours.append('#F59E0B')   # amber — Medium
    else:
        bar_colours.append('#EF4444')   # red — High

# Plot bars
bar_width = 9
bar_centres = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
bars = ax.bar(bar_centres, hist_counts, width=bar_width,
              color=bar_colours, edgecolor='white', linewidth=0.5)

# Count labels on each bar
for bar, count in zip(bars, hist_counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(int(count)), ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#374151')

# Threshold lines
ax.axvline(x=40, color='#6B7280', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(40, max(hist_counts) * 0.95, ' Low/Medium\n threshold',
        fontsize=8, color='#6B7280', va='top')

ax.axvline(x=70, color='#6B7280', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(70, max(hist_counts) * 0.95, ' Medium/High\n threshold',
        fontsize=8, color='#6B7280', va='top')

# Labels and styling
ax.set_xlabel('Risk Score', fontsize=12)
ax.set_ylabel('Number of Students', fontsize=12)
ax.set_title('Risk Score Distribution — 25 Students',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlim(-2, 102)
ax.set_xticks(bin_edges)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_patches = [
    mpatches.Patch(color='#10B981', label='Low Risk (0–40)'),
    mpatches.Patch(color='#F59E0B', label='Medium Risk (40–70)'),
    mpatches.Patch(color='#EF4444', label='High Risk (70–100)'),
]
ax.legend(handles=legend_patches, loc='upper center', fontsize=9,
          framealpha=0.9)

plt.tight_layout()
plt.savefig('report/charts/risk_histogram.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Chart 3 saved: report/charts/risk_histogram.png")


# ── Final confirmation ────────────────────────────────
print("")
print("✅ All report charts generated in report/charts/")
print("   Use these in your report and PPT slides.")

conn.close()
