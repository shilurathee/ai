import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — required before pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
from io import BytesIO
from model.predict import (
    predict_all_students,
    predict_student,
    extract_features,
    _rule_based_score,
    _risk_category,
    _recommendation,
    _top_3_factors,
    _load_metrics,
    FEATURE_COLS,
)

# ── Page config — must be first Streamlit call ────────
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════
# SECTION 2: CONSTANTS AND PATHS
# ═══════════════════════════════════════════════════════

DB_PATH      = 'dropout_predictor.db'
METRICS_PATH = 'model/metrics.pkl'
CHART_PATH   = 'model/feature_importance.png'

# Colour palette — used consistently across all screens
COLOUR_HIGH   = '#EF4444'   # red
COLOUR_MEDIUM = '#F59E0B'   # amber
COLOUR_LOW    = '#10B981'   # green
COLOUR_BLUE   = '#3B82F6'   # blue (neutral/info)

# Map category string to hex colour
CATEGORY_COLOURS = {
    'High':   COLOUR_HIGH,
    'Medium': COLOUR_MEDIUM,
    'Low':    COLOUR_LOW,
}


# ═══════════════════════════════════════════════════════
# SECTION 3: CACHED DATA LOADING
# ═══════════════════════════════════════════════════════

@st.cache_resource
def get_db_connection():
    """Open and return a persistent SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


@st.cache_data
def load_all_predictions():
    """
    Load predictions for all 25 students.
    Returns a list of result dicts sorted by risk_score descending.
    Cached so it only runs once per session.
    """
    conn = get_db_connection()
    return predict_all_students(conn)


@st.cache_data
def load_metrics():
    """Load the saved model metrics dict from disk."""
    return _load_metrics()


# ═══════════════════════════════════════════════════════
# SECTION 5: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════

def badge_html(category: str) -> str:
    """Return an HTML pill badge coloured by risk category."""
    colour = CATEGORY_COLOURS.get(category, '#6B7280')
    return (
        '<span style="background-color:{}; color:white; '
        'padding:3px 10px; border-radius:12px; font-size:13px; '
        'font-weight:600;">{}</span>'.format(colour, category)
    )


def risk_gauge(score: float, category: str):
    """
    Draw a semicircle gauge chart showing the student's risk score (0–100).
    Returns a BytesIO PNG buffer for st.image().
    """
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.set_aspect('equal')
    ax.axis('off')

    # Grey background semicircle arc (the track)
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#E5E7EB', linewidth=18)

    # Coloured filled arc from π down to the score position
    score_angle = np.pi - (score / 100) * np.pi
    theta2 = np.linspace(np.pi, score_angle, 200)
    ax.plot(np.cos(theta2), np.sin(theta2),
            color=CATEGORY_COLOURS.get(category, '#6B7280'), linewidth=18)

    # Centre text — risk score value
    ax.text(0, -0.15, '{:.0f}'.format(score),
            ha='center', va='center',
            fontsize=36, fontweight='bold',
            color=CATEGORY_COLOURS.get(category, '#6B7280'))
    # Centre label
    ax.text(0, -0.45, 'Risk Score',
            ha='center', va='center', fontsize=11, color='#6B7280')

    # Low / Medium / High labels at gauge edges
    ax.text(-1.05, -0.05, 'Low', ha='center', fontsize=8, color='#6B7280')
    ax.text(0, -0.65, 'Medium', ha='center', fontsize=8, color='#6B7280')
    ax.text(1.05, -0.05, 'High', ha='center', fontsize=8, color='#6B7280')

    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return buf


def marks_bar_chart(features: dict):
    """
    Draw a horizontal bar chart of the student's 8 feature values.
    Returns a BytesIO PNG buffer for st.image().
    """
    labels = [
        'Avg Marks', 'Courses Enrolled', 'Zero-Mark Courses',
        'Missing Assessments', 'Failing Courses',
        'Lowest Course Score', 'Marks Trend', 'Completion Rate',
    ]
    values = [features[col] for col in FEATURE_COLS]

    # Colour each bar based on whether the value is concerning
    colours = []
    for col, val in zip(FEATURE_COLS, values):
        if col in ('avg_marks', 'lowest_course_score',
                    'completion_rate') and val < 0.5:
            colours.append(COLOUR_HIGH)
        elif col in ('assessments_missing', 'grade_f_count',
                      'courses_with_zero_marks') and val >= 2:
            colours.append(COLOUR_MEDIUM)
        else:
            colours.append(COLOUR_BLUE)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colours, edgecolor='white',
                   height=0.6)
    ax.set_xlabel('Feature Value')
    ax.set_title('Student Feature Profile', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Value labels on each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                '{:.2f}'.format(val), va='center', fontsize=8)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════
# SECTION 6: SCREEN 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════

def render_overview():
    st.title("📊 Student Dropout Risk Overview")
    st.markdown("AI-generated risk scores for all enrolled students.")
    st.markdown("---")

    # Load cached predictions
    all_results = load_all_predictions()

    # Count students in each risk category
    high_count   = sum(1 for r in all_results if r['risk_category'] == 'High')
    medium_count = sum(1 for r in all_results if r['risk_category'] == 'Medium')
    low_count    = sum(1 for r in all_results if r['risk_category'] == 'Low')

    # ── 4 metric cards at top with custom styling ─────
    # Custom CSS to style metric cards
    st.markdown("""
        <style>
          div[data-testid="metric-container"] {
            background-color: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 16px;
          }
        </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(all_results))
    c2.metric("🔴 High Risk", high_count,
              help="Students needing immediate intervention")
    c3.metric("🟡 Medium Risk", medium_count,
              help="Students to monitor closely")
    c4.metric("🟢 Low Risk", low_count,
              help="Students on track")

    st.markdown("")

    # ── Risk table (left) + Donut chart (right) ──────
    col_left, col_right = st.columns([2, 1])

    with col_right:
        # Risk distribution donut chart
        st.markdown("#### 🥧 Risk Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sizes   = [high_count, medium_count, low_count]
        labels  = [
            'High ({})'.format(high_count),
            'Medium ({})'.format(medium_count),
            'Low ({})'.format(low_count),
        ]
        colours = [COLOUR_HIGH, COLOUR_MEDIUM, COLOUR_LOW]

        # Only include non-zero slices to avoid matplotlib warnings
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colours) if s > 0]
        if non_zero:
            nz_sizes, nz_labels, nz_colours = zip(*non_zero)
            ax.pie(nz_sizes, labels=nz_labels, colors=nz_colours,
                   autopct='%1.1f%%', startangle=90,
                   wedgeprops=dict(width=0.55, edgecolor='white'),
                   textprops={'fontsize': 10})
        ax.set_title('Risk Category Distribution', fontweight='bold', pad=10)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        buf.seek(0)
        # Render donut chart image
        st.image(buf, use_container_width=True)

    with col_left:
        # Student risk HTML table with coloured badges
        st.markdown("#### 📋 Student Risk Table")

        # Build HTML table header — NO leading whitespace so Streamlit renders it as HTML
        table_rows = []
        for i, r in enumerate(all_results):
            bg = '#F9FAFB' if i % 2 == 0 else '#FFFFFF'
            avg_pct = round(r['features']['avg_marks'] * 100, 1)
            comp_pct = round(r['features']['completion_rate'] * 100, 1)
            missing = r['features']['assessments_missing']
            badge = badge_html(r['risk_category'])
            table_rows.append(
                '<tr style="background:{}; border-bottom:1px solid #E5E7EB;">'
                '<td style="padding:8px 12px; font-weight:500;">{}</td>'
                '<td style="text-align:center; padding:8px 12px;">{}</td>'
                '<td style="text-align:center; padding:8px 12px;">{}</td>'
                '<td style="text-align:center; padding:8px 12px;">{}</td>'
                '<td style="text-align:center; padding:8px 12px;">{}</td>'
                '<td style="text-align:center; padding:8px 12px;">{}</td>'
                '</tr>'.format(bg, r['student_name'], r['risk_score'],
                               badge, avg_pct, missing, comp_pct)
            )

        html = (
            '<div style="overflow-x:auto;">'
            '<table style="width:100%; border-collapse:collapse; font-family:sans-serif; font-size:14px;">'
            '<thead>'
            '<tr style="background:#F3F4F6; border-bottom:2px solid #D1D5DB;">'
            '<th style="text-align:left; padding:10px 12px;">Student Name</th>'
            '<th style="text-align:center; padding:10px 12px;">Risk Score</th>'
            '<th style="text-align:center; padding:10px 12px;">Category</th>'
            '<th style="text-align:center; padding:10px 12px;">Avg Marks %</th>'
            '<th style="text-align:center; padding:10px 12px;">Missing Asmts</th>'
            '<th style="text-align:center; padding:10px 12px;">Completion %</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            + ''.join(table_rows)
            + '</tbody></table></div>'
        )

        # Render the styled HTML table
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Student selector to jump to Student Detail screen ──
    selected_name = st.selectbox(
        "Select a student to view their full detail →",
        options=[r['student_name'] for r in all_results],
    )
    if st.button("View Student Detail →"):
        st.session_state['selected_student'] = selected_name
        st.session_state['page'] = "👤 Student Detail"
        st.rerun()


# ═══════════════════════════════════════════════════════
# SECTION 7: SCREEN 2 — STUDENT DETAIL
# ═══════════════════════════════════════════════════════

def render_student_detail():
    all_results = load_all_predictions()
    names = [r['student_name'] for r in all_results]

    # Check if a student was pre-selected from Screen 1
    default_name = st.session_state.get('selected_student', names[0])
    default_idx = names.index(default_name) if default_name in names else 0

    chosen_name = st.selectbox("Select Student", options=names,
                               index=default_idx)

    # Find chosen student's result dict
    result = next(r for r in all_results if r['student_name'] == chosen_name)

    # ── Page header ──────────────────────────────────
    st.title("👤 {}".format(result['student_name']))
    st.markdown(
        "**Student ID:** {}  |  **Method:** {}".format(
            result['student_id'], result['method']
        )
    )
    st.markdown("---")

    # ── Top section: gauge + risk factors side by side ──
    col_gauge, col_factors = st.columns([1, 2])

    with col_gauge:
        # Semicircle gauge chart
        buf = risk_gauge(result['risk_score'], result['risk_category'])
        st.image(buf, use_container_width=True)
        # Coloured category badge centred under gauge
        st.markdown(
            "<div style='text-align:center'>{}</div>".format(
                badge_html(result['risk_category'])),
            unsafe_allow_html=True
        )

    with col_factors:
        # Top risk factors as styled warning cards
        st.markdown("#### ⚠️ Top Risk Factors")
        for factor in result['top_factors']:
            if factor == "No major risk factors detected":
                st.success("✅ " + factor)
            else:
                # Amber warning card with left border accent
                st.markdown(
                    '<div style="background:#FEF3C7; border-left:4px solid #F59E0B;'
                    ' padding:10px 14px; margin-bottom:8px;'
                    ' border-radius:6px; font-size:14px;">'
                    '⚡ {}</div>'.format(factor),
                    unsafe_allow_html=True
                )

        # Recommendation card coloured by risk level
        st.markdown("#### 📋 Recommended Action")
        rec_colour_map = {
            'High': '#FEE2E2', 'Medium': '#FEF3C7', 'Low': '#D1FAE5'
        }
        rec_bg = rec_colour_map.get(result['risk_category'], '#F3F4F6')
        # Recommendation box with background matching risk severity
        st.markdown(
            '<div style="background:{}; padding:12px 16px;'
            ' border-radius:8px; font-size:14px;">'
            '{}</div>'.format(rec_bg, result['recommendation']),
            unsafe_allow_html=True
        )

    # ── Feature profile bar chart ────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Student Feature Profile")
    buf2 = marks_bar_chart(result['features'])
    st.image(buf2, use_container_width=True)

    # ── Raw feature values in an expander ────────────
    with st.expander("🔍 View Raw Feature Values"):
        desc_map = {
            'avg_marks':               'Mean score across all assessments (0–1)',
            'courses_enrolled':        'Number of courses enrolled',
            'courses_with_zero_marks': 'Courses where avg mark = 0',
            'assessments_missing':     'Assessments not submitted',
            'grade_f_count':           'Courses failed (below 40%)',
            'lowest_course_score':     'Worst course average score (0–1)',
            'marks_trend':             'Slope of marks over time (+ = improving)',
            'completion_rate':         'Fraction of assessments submitted (0–1)',
        }
        feature_df = pd.DataFrame([{
            'Feature':     col,
            'Value':       result['features'][col],
            'Description': desc_map[col],
        } for col in FEATURE_COLS])
        st.dataframe(feature_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# SECTION 8: SCREEN 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════

def render_model_insights():
    st.title("🤖 Model Insights")
    st.markdown(
        "Details about the Random Forest model trained in Phase 3. "
        "Understand which features drive dropout predictions most."
    )
    st.markdown("---")

    # Load metrics
    metrics = load_metrics()

    if not metrics:
        st.warning(
            "Model not trained yet. Run `python model/train.py` first.")
        st.stop()

    # ── 4 accuracy metric cards ──────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "{}%".format(metrics['accuracy']))
    c2.metric("Precision", "{}%".format(metrics['precision']),
              help="Of students flagged as dropout, how many truly were?")
    c3.metric("Recall", "{}%".format(metrics['recall']),
              help="Of all actual dropouts, how many did we catch?")
    c4.metric("F1 Score", "{}%".format(metrics['f1']),
              help="Harmonic mean of Precision and Recall")

    st.markdown("")

    # ── Feature importance chart from trained model ──
    st.markdown("#### 📊 Feature Importance Chart")
    st.markdown(
        "The chart below shows which features the Random Forest uses "
        "most to decide if a student is at risk. Taller bars = "
        "more influence on the prediction."
    )
    if os.path.exists(CHART_PATH):
        st.image(CHART_PATH, use_container_width=True)
    else:
        st.error("Feature importance chart not found. "
                 "Run `python model/train.py` to generate it.")

    # ── Confusion matrix heatmap ─────────────────────
    st.markdown("#### 🔢 Confusion Matrix")
    st.markdown(
        "Shows how well the model performed on the 100-row test set. "
        "Rows = actual labels, Columns = predicted labels."
    )

    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted No', 'Predicted Yes'])
    ax.set_yticklabels(['Actual No', 'Actual Yes'])
    plt.colorbar(im, ax=ax)

    # Cell value labels inside the heatmap
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title('Confusion Matrix — Test Set', fontweight='bold', pad=10)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    # Render confusion matrix heatmap
    st.image(buf, width=400)

    # ── Risk distribution pie across all 25 real students ──
    st.markdown("#### 🥧 Risk Distribution — All Students")
    all_results = load_all_predictions()
    high   = sum(1 for r in all_results if r['risk_category'] == 'High')
    medium = sum(1 for r in all_results if r['risk_category'] == 'Medium')
    low    = sum(1 for r in all_results if r['risk_category'] == 'Low')

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sizes  = [high, medium, low]
    labels = [
        'High ({})'.format(high),
        'Medium ({})'.format(medium),
        'Low ({})'.format(low),
    ]
    colours = [COLOUR_HIGH, COLOUR_MEDIUM, COLOUR_LOW]

    # Only include non-zero slices
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colours) if s > 0]
    if non_zero:
        nz_sizes, nz_labels, nz_colours = zip(*non_zero)
        ax2.pie(nz_sizes, labels=nz_labels, colors=nz_colours,
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.55, edgecolor='white'),
                textprops={'fontsize': 10})
    ax2.set_title('Risk Category Distribution\n(25 Real DB Students)',
                  fontweight='bold')
    plt.tight_layout()

    buf2 = BytesIO()
    plt.savefig(buf2, format='png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close()
    buf2.seek(0)
    # Render risk distribution donut chart
    st.image(buf2, use_container_width=False, width=450)

    # ── Technical model details in an expander ───────
    with st.expander("ℹ️ Technical Model Details"):
        # Model hyperparameters and training configuration table
        st.markdown("""
          | Parameter         | Value                    |
          |-------------------|--------------------------|
          | Algorithm         | Random Forest Classifier |
          | Trees             | 100                      |
          | Max Depth         | 8                        |
          | Class Weight      | Balanced                 |
          | Training Rows     | {}                       |
          | Test Rows         | {}                       |
          | Training CSV      | data/training_data.csv   |
          | Dropout Rate (CSV)| {}%                      |
          | Random Seed       | 42                       |
        """.format(
            metrics.get('train_size', 400),
            metrics.get('test_size', 100),
            metrics.get('dropout_rate_pct', '~20'),
        ))


# ═══════════════════════════════════════════════════════
# SECTION 9: SCREEN 4 — SIMULATE STUDENT
# ═══════════════════════════════════════════════════════

def render_simulate():
    st.title("🧪 Simulate a Student")
    st.markdown(
        "Enter hypothetical student marks below. Click **Predict Risk** "
        "to see the AI's dropout risk prediction in real time. "
        "No database needed — works entirely from the input values."
    )
    st.markdown("---")

    # ── Input form with 2-column layout ──────────────
    with st.form(key='simulate_form'):
        form_left, form_right = st.columns([1, 1])

        with form_left:
            avg_marks = st.slider(
                "Average Marks (as fraction)",
                min_value=0.0, max_value=1.0, value=0.6, step=0.01,
                help="e.g. 0.65 means 65% average across all assessments"
            )
            courses_enrolled = st.number_input(
                "Courses Enrolled", min_value=1, max_value=6,
                value=4, step=1
            )
            courses_with_zero_marks = st.number_input(
                "Courses with Zero Marks", min_value=0, max_value=6,
                value=0, step=1
            )
            assessments_missing = st.number_input(
                "Assessments Missing", min_value=0, max_value=20,
                value=2, step=1
            )

        with form_right:
            grade_f_count = st.number_input(
                "Courses Failed (below 40%)", min_value=0, max_value=6,
                value=1, step=1
            )
            lowest_course_score = st.slider(
                "Lowest Course Score (fraction)",
                min_value=0.0, max_value=1.0, value=0.4, step=0.01,
                help="Worst-performing course average score"
            )
            marks_trend = st.slider(
                "Marks Trend (slope)",
                min_value=-1.0, max_value=1.0, value=0.0, step=0.01,
                help="Negative = declining, Positive = improving"
            )
            completion_rate = st.slider(
                "Completion Rate (fraction)",
                min_value=0.0, max_value=1.0, value=0.8, step=0.01,
                help="Fraction of assessments submitted"
            )

        # Submit button
        submitted = st.form_submit_button(
            "🔮 Predict Risk",
            use_container_width=True,
            type="primary",
        )

    # ── Prediction on form submit ────────────────────
    if submitted:
        # Build features dict from user inputs
        sim_features = {
            'avg_marks':               avg_marks,
            'courses_enrolled':        courses_enrolled,
            'courses_with_zero_marks': courses_with_zero_marks,
            'assessments_missing':     assessments_missing,
            'grade_f_count':           grade_f_count,
            'lowest_course_score':     lowest_course_score,
            'marks_trend':             marks_trend,
            'completion_rate':         completion_rate,
        }

        # Try ML model first, fallback to rule-based
        try:
            with open('model/rf_model.pkl', 'rb') as f:
                model = pickle.load(f)
            X = np.array([[sim_features[c] for c in FEATURE_COLS]])
            prob = model.predict_proba(X)[0][1]
            risk_score = round(float(prob) * 100, 1)
            method_used = "🤖 ML Model (Random Forest)"
        except Exception:
            risk_score  = _rule_based_score(sim_features)
            method_used = "📐 Rule-based Fallback"

        category       = _risk_category(risk_score)
        metrics        = load_metrics()
        top_factors    = _top_3_factors(sim_features, metrics)
        recommendation = _recommendation(category)

        # ── Show simulation results ──────────────────
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_left, res_right = st.columns([1, 2])

        with res_left:
            # Gauge chart for simulated student
            buf = risk_gauge(risk_score, category)
            st.image(buf, use_container_width=True)
            # Category badge centred under gauge
            st.markdown(
                "<div style='text-align:center; margin-top:-10px'>"
                "{}</div>".format(badge_html(category)),
                unsafe_allow_html=True
            )
            # Method used label
            st.markdown(
                "<div style='text-align:center; color:#6B7280; font-size:12px; "
                "margin-top:4px'>Method: {}</div>".format(method_used),
                unsafe_allow_html=True
            )

        with res_right:
            # Risk factors section
            st.markdown("#### ⚠️ Risk Factors Identified")
            if top_factors == ["No major risk factors detected"]:
                st.success("No significant risk factors detected for this profile.")
            else:
                for factor in top_factors:
                    # Amber warning card per risk factor
                    st.markdown(
                        '<div style="background:#FEF3C7; border-left:4px solid #F59E0B;'
                        ' padding:10px 14px; margin-bottom:8px;'
                        ' border-radius:6px; font-size:14px;">'
                        '⚡ {}</div>'.format(factor),
                        unsafe_allow_html=True
                    )

            # Recommendation card
            st.markdown("#### 📋 Recommended Action")
            rec_colour_map = {
                'High': '#FEE2E2', 'Medium': '#FEF3C7', 'Low': '#D1FAE5'
            }
            rec_bg = rec_colour_map.get(category, '#F3F4F6')
            # Recommendation box styled by severity
            st.markdown(
                '<div style="background:{}; padding:12px 16px;'
                ' border-radius:8px; font-size:14px; margin-top:4px;">'
                '{}</div>'.format(rec_bg, recommendation),
                unsafe_allow_html=True
            )

    # ── Tip note below the result ────────────────────
    st.markdown("---")
    st.info(
        "💡 **Tip:** Adjust the sliders above and click "
        "**Predict Risk** again to see how different inputs "
        "change the dropout risk score in real time."
    )


# ═══════════════════════════════════════════════════════
# SECTION 4: SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎓 Dropout Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        options=[
            "📊 Overview Dashboard",
            "👤 Student Detail",
            "🤖 Model Insights",
            "🧪 Simulate Student",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### About")
    st.markdown(
        "AI-powered early warning system "
        "for student dropout risk. "
        "Built with Random Forest + Streamlit."
    )


# ═══════════════════════════════════════════════════════
# SECTION 10: ROUTE BETWEEN SCREENS
# ═══════════════════════════════════════════════════════

if "Overview" in page:
    render_overview()
elif "Simulate" in page:
    render_simulate()
elif "Student" in page:
    render_student_detail()
elif "Insights" in page:
    render_model_insights()
