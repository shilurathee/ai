import sqlite3
import numpy as np
import pickle
import os

# ── Constants for model loading ───────────────────────
MODEL_PATH = 'model/rf_model.pkl'
METRICS_PATH = 'model/metrics.pkl'
FEATURE_COLS = [
    'avg_marks', 'courses_enrolled', 'courses_with_zero_marks',
    'assessments_missing', 'grade_f_count', 'lowest_course_score',
    'marks_trend', 'completion_rate'
]


# ═══════════════════════════════════════════════════════
# PHASE 2 FUNCTIONS (unchanged)
# ═══════════════════════════════════════════════════════

def extract_features(student_id: int, db_conn: sqlite3.Connection) -> dict:
    """
    Query the SQLite database and compute all 8 dropout risk features
    for a given student. Returns a dict with exactly these 8 keys:
      avg_marks, courses_enrolled, courses_with_zero_marks,
      assessments_missing, grade_f_count, lowest_course_score,
      marks_trend, completion_rate
    Returns None if student_id does not exist in the database.
    """

    # 1. Validate student exists
    row = db_conn.execute(
        "SELECT id FROM students WHERE id = ?", (student_id,)
    ).fetchone()
    if row is None:
        return None

    # 2. avg_marks — mean of (obtained/max) across all non-NULL assessments
    rows = db_conn.execute(
        """SELECT a.obtained_marks, a.max_marks
           FROM assessments a
           JOIN enrollments e ON a.enrollment_id = e.id
           WHERE e.student_id = ?""",
        (student_id,)
    ).fetchall()

    ratios = []
    for obtained, max_m in rows:
        if obtained is not None:
            ratios.append(obtained / max_m)
    avg_marks = sum(ratios) / len(ratios) if ratios else 0.0

    # 3. courses_enrolled — total number of enrollments for this student
    courses_enrolled = db_conn.execute(
        "SELECT COUNT(*) FROM enrollments WHERE student_id = ?",
        (student_id,)
    ).fetchone()[0]

    # 4. courses_with_zero_marks — courses where avg obtained (NULL→0) is exactly 0
    per_course_avg = db_conn.execute(
        """SELECT e.course_id,
                  AVG(COALESCE(a.obtained_marks, 0)) as avg_obtained
           FROM enrollments e
           JOIN assessments a ON a.enrollment_id = e.id
           WHERE e.student_id = ?
           GROUP BY e.course_id""",
        (student_id,)
    ).fetchall()

    courses_with_zero_marks = sum(1 for _, avg_o in per_course_avg if avg_o == 0)

    # 5. assessments_missing — count of assessments where obtained_marks IS NULL
    assessments_missing = db_conn.execute(
        """SELECT COUNT(*) FROM assessments a
           JOIN enrollments e ON a.enrollment_id = e.id
           WHERE e.student_id = ? AND a.obtained_marks IS NULL""",
        (student_id,)
    ).fetchone()[0]

    # 6. grade_f_count — courses where sum(obtained)/sum(max) < 0.40
    per_course_score = db_conn.execute(
        """SELECT e.course_id,
                  SUM(COALESCE(a.obtained_marks, 0)) * 1.0 / SUM(a.max_marks)
                    as score_ratio
           FROM enrollments e
           JOIN assessments a ON a.enrollment_id = e.id
           WHERE e.student_id = ?
           GROUP BY e.course_id""",
        (student_id,)
    ).fetchall()

    grade_f_count = sum(1 for _, ratio in per_course_score if ratio < 0.40)

    # 7. lowest_course_score — minimum per-course score ratio
    if per_course_score:
        lowest_course_score = min(ratio for _, ratio in per_course_score)
    else:
        lowest_course_score = 0.0

    # 8. marks_trend — slope of obtained_marks over time (NULL→0)
    marks_rows = db_conn.execute(
        """SELECT a.obtained_marks FROM assessments a
           JOIN enrollments e ON a.enrollment_id = e.id
           WHERE e.student_id = ?
           ORDER BY a.id ASC""",
        (student_id,)
    ).fetchall()

    marks = [m[0] if m[0] is not None else 0 for m in marks_rows]
    if len(marks) < 2:
        marks_trend = 0.0
    else:
        marks_trend = np.polyfit(range(len(marks)), marks, 1)[0]

    # 9. completion_rate — fraction of assessments that were submitted
    rate_row = db_conn.execute(
        """SELECT
             COUNT(CASE WHEN a.obtained_marks IS NOT NULL THEN 1 END) * 1.0
               / COUNT(*) as rate
           FROM assessments a
           JOIN enrollments e ON a.enrollment_id = e.id
           WHERE e.student_id = ?""",
        (student_id,)
    ).fetchone()

    # Handle edge case of zero assessments
    total_assessments = db_conn.execute(
        """SELECT COUNT(*) FROM assessments a
           JOIN enrollments e ON a.enrollment_id = e.id
           WHERE e.student_id = ?""",
        (student_id,)
    ).fetchone()[0]

    completion_rate = rate_row[0] if total_assessments > 0 else 0.0

    # Return all 8 features as a dict with proper rounding
    return {
        'avg_marks':               round(avg_marks, 4),
        'courses_enrolled':        int(courses_enrolled),
        'courses_with_zero_marks': int(courses_with_zero_marks),
        'assessments_missing':     int(assessments_missing),
        'grade_f_count':           int(grade_f_count),
        'lowest_course_score':     round(lowest_course_score, 4),
        'marks_trend':             round(marks_trend, 4),
        'completion_rate':         round(completion_rate, 4),
    }


def get_all_student_ids(db_conn: sqlite3.Connection) -> list:
    """Return a list of all student ids in the database."""
    rows = db_conn.execute("SELECT id FROM students ORDER BY id").fetchall()
    return [r[0] for r in rows]


def get_student_name(student_id: int, db_conn: sqlite3.Connection) -> str:
    """Return the name of a student by id, or 'Unknown' if not found."""
    row = db_conn.execute(
        "SELECT name FROM students WHERE id = ?", (student_id,)
    ).fetchone()
    return row[0] if row else 'Unknown'


# ═══════════════════════════════════════════════════════
# PHASE 3 FUNCTIONS (new)
# ═══════════════════════════════════════════════════════

def _load_model():
    """
    Load the trained Random Forest model from disk.
    Returns the model object, or None if the file does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def _load_metrics() -> dict:
    """
    Load the saved metrics dict from disk.
    Returns an empty dict if the file does not exist.
    """
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, 'rb') as f:
        return pickle.load(f)


def _rule_based_score(features: dict) -> float:
    """
    Compute a dropout risk score (0–100) using a weighted formula.
    Used as fallback when rf_model.pkl does not exist.
    Fully explainable — useful for project reports.

    Formula:
      score = (
        (1 - avg_marks)           * 0.30 +
        (assessments_missing/10)  * 0.25 +
        (grade_f_count/5)         * 0.25 +
        (1 - completion_rate)     * 0.20
      ) * 100

    Clip result to range [0, 100].
    """
    score = (
        (1 - features['avg_marks'])                     * 0.30 +
        (min(features['assessments_missing'], 10) / 10) * 0.25 +
        (min(features['grade_f_count'], 5) / 5)         * 0.25 +
        (1 - features['completion_rate'])                * 0.20
    ) * 100
    return round(float(np.clip(score, 0, 100)), 1)


def _risk_category(score: float) -> str:
    """
    Map a numeric risk score to a category label.
      score < 40   → 'Low'
      40 ≤ score < 70 → 'Medium'
      score ≥ 70   → 'High'
    """
    if score >= 70:
        return 'High'
    elif score >= 40:
        return 'Medium'
    else:
        return 'Low'


def _top_3_factors(features: dict, metrics: dict) -> list:
    """
    Identify the 3 most concerning factors for this specific student.
    Uses feature importances from the trained model (from metrics dict)
    to rank which features matter most, then maps poor values to
    human-readable warning strings.

    Returns a list of up to 3 strings.
    """

    # Define what counts as a "bad" value per feature,
    # and the warning message to show if it is bad.
    rules = [
        (
            'avg_marks',
            lambda v: v < 0.50,
            lambda v: "Very low average marks ({}%)".format(round(v * 100))
        ),
        (
            'assessments_missing',
            lambda v: v >= 3,
            lambda v: "Missing {} assessment(s)".format(int(v))
        ),
        (
            'grade_f_count',
            lambda v: v >= 2,
            lambda v: "Failing {} course(s) (below 40%)".format(int(v))
        ),
        (
            'completion_rate',
            lambda v: v < 0.70,
            lambda v: "Low completion rate ({}%)".format(round(v * 100))
        ),
        (
            'courses_with_zero_marks',
            lambda v: v >= 1,
            lambda v: "Zero marks in {} course(s)".format(int(v))
        ),
        (
            'marks_trend',
            lambda v: v < -0.1,
            lambda v: "Declining marks trend (slope: {})".format(round(v, 2))
        ),
        (
            'lowest_course_score',
            lambda v: v < 0.35,
            lambda v: "Critically low score in weakest course ({}%)".format(
                round(v * 100))
        ),
    ]

    # Get feature importance weights from metrics (fallback to equal weight)
    importances = {}
    if metrics and 'feature_importances' in metrics:
        for name, imp in zip(metrics['feature_names'],
                             metrics['feature_importances']):
            importances[name] = imp
    else:
        for name in FEATURE_COLS:
            importances[name] = 1.0

    # Collect all bad factors with their importance weight
    bad_factors = []
    for (feat, is_bad, msg_fn) in rules:
        val = features.get(feat, 0)
        if is_bad(val):
            bad_factors.append((importances.get(feat, 0), msg_fn(val)))

    # Sort by importance descending, return top 3 messages
    bad_factors.sort(key=lambda x: x[0], reverse=True)
    top = [msg for _, msg in bad_factors[:3]]

    if not top:
        top = ["No major risk factors detected"]

    return top


def _recommendation(category: str) -> str:
    """
    Return a plain-English action recommendation based on risk category.
    """
    recs = {
        'High':   ("⚠️ Immediate academic counselling advised. "
                   "Contact student and guardian within 48 hours."),
        'Medium': ("📋 Schedule a check-in meeting this week. "
                   "Review attendance and assignment submissions."),
        'Low':    ("✅ Student is on track. Continue routine monitoring "
                   "and encourage participation."),
    }
    return recs.get(category, recs['Low'])


def predict_student(student_id: int, db_conn: sqlite3.Connection) -> dict:
    """
    Full prediction pipeline for a single student.

    Steps:
      1. Extract features from the database via extract_features()
      2. Try to use the Random Forest model for prediction
      3. Fall back to _rule_based_score() if model not found
      4. Compute risk category, top factors, and recommendation
      5. Return a comprehensive result dict

    Returns None if student_id does not exist.

    Return dict structure:
      {
        'student_id':     int,
        'student_name':   str,
        'features':       dict (the 8 features),
        'risk_score':     float (0–100),
        'risk_category':  str ('Low', 'Medium', 'High'),
        'method':         str ('Random Forest' or 'Rule-based'),
        'top_factors':    list[str] (up to 3 human-readable warnings),
        'recommendation': str,
        'dropout_prob':   float (0.0–1.0, from RF; or score/100 for rule-based),
      }
    """

    # Step 1: Extract features from the database
    features = extract_features(student_id, db_conn)
    if features is None:
        return None

    # Step 2: Get student name
    name = get_student_name(student_id, db_conn)

    # Step 3: Try RF model, fall back to rule-based
    model = _load_model()
    metrics = _load_metrics()

    if model is not None:
        # Use RF model for prediction
        feature_values = np.array([[features[col] for col in FEATURE_COLS]])
        dropout_prob = float(model.predict_proba(feature_values)[0][1])
        risk_score = round(dropout_prob * 100, 1)
        method = 'Random Forest'
    else:
        # Fallback to rule-based scoring
        risk_score = _rule_based_score(features)
        dropout_prob = round(risk_score / 100, 4)
        method = 'Rule-based'

    # Step 4: Determine category, factors, and recommendation
    category = _risk_category(risk_score)
    top_factors = _top_3_factors(features, metrics)
    rec = _recommendation(category)

    # Step 5: Return comprehensive result dict
    return {
        'student_id':     student_id,
        'student_name':   name,
        'features':       features,
        'risk_score':     risk_score,
        'risk_category':  category,
        'method':         method,
        'top_factors':    top_factors,
        'recommendation': rec,
        'dropout_prob':   round(dropout_prob, 4),
    }


def predict_all_students(db_conn: sqlite3.Connection) -> list:
    """
    Run predict_student() for every student in the database.
    Returns a list of result dicts sorted by risk_score descending
    (highest risk first).
    """
    all_ids = get_all_student_ids(db_conn)
    results = []
    for sid in all_ids:
        result = predict_student(sid, db_conn)
        if result is not None:
            results.append(result)
    # Sort by risk score descending — highest risk students first
    results.sort(key=lambda r: r['risk_score'], reverse=True)
    return results
