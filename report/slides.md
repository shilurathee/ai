# Student Dropout Risk Predictor — PPT Slide Content
## Complete 10-Slide Presentation

---

## SLIDE 1: TITLE SLIDE

**Title:** Student Dropout Risk Predictor
**Subtitle:** An AI-Powered Early Warning System for Educators
**Line 3:** [Your Name] | [College Name] | [Date]

**Layout:** Full-screen dark blue background (#1E3A5F), white text, graduation cap emoji 🎓 centred above the title. Clean, minimal — no bullets, no charts.

**Visual:** No chart needed. Clean title layout only.

**Speaker Notes:**
"Good morning. Today I will be presenting our project — a Student Dropout Risk Predictor built using Python, scikit-learn, and Streamlit. The goal of this system is to help educators identify students who are at risk of dropping out before it is too late to intervene. I will walk you through the problem we are solving, the AI model we built, our results, and then finish with a live demonstration of the working dashboard."

---

## SLIDE 2: PROBLEM STATEMENT

**Title:** The Dropout Problem

**Bullets:**
- Globally, 30–40% of enrolled students do not complete their degree
- Late detection = missed opportunity for intervention
- Manual grade reviews are slow, subjective, and reactive
- Educators need an early warning system — not a post-dropout report

**Call-out box** (coloured text box, #FEF3C7 background, #92400E text):
> "Every dropout is a student who could have been saved with earlier, targeted support."

**Layout:** 2 columns — bullets on the left, large statistic on the right. Right side shows "30–40%" in 96pt font, bold, red (#EF4444), with caption below in 16pt: "of students don't finish their degree".

**Visual:** Red warning icon or a downward-trending student enrollment chart as a placeholder image on the right side behind the statistic.

**Speaker Notes:**
"The core problem we are solving is student dropout — a challenge that affects universities worldwide. Studies suggest that between 30 and 40 percent of enrolled students do not complete their degrees. The tragedy is that many of these students could have been supported if faculty had known earlier that they were struggling. Current processes rely on manual grade reviews which are slow and happen after the student has already fallen behind. Our system flips this — it proactively scores every student and flags those at risk before they disengage completely."

---

## SLIDE 3: SOLUTION OVERVIEW

**Title:** Our Solution — An AI Early Warning System

**Bullets:**
- Extracts 8 risk features per student from a structured database
- Trained a Random Forest ML model on 500 synthetic student profiles
- Predicts dropout probability as a 0–100 risk score per student
- Presents results on a Streamlit web dashboard for educators
- Fallback rule-based scoring works even without the ML model

**Layout:** Centre a 5-step horizontal flow diagram below the bullets:

```
[Database] → [Feature Extraction] → [ML Model] → [Risk Score] → [Dashboard]
   📁              🔧                   🌲            📊            🖥️
```

Use arrows between each step. Colour each box differently:
- Database: Blue (#3B82F6)
- Feature Extraction: Purple (#8B5CF6)
- ML Model: Green (#10B981)
- Risk Score: Amber (#F59E0B)
- Dashboard: Red (#EF4444)

**Visual:** The 5-step flow diagram described above, drawn with PowerPoint shapes.

**Speaker Notes:**
"Our solution works in five steps. First, student data — marks, courses, assessments — is stored in a structured SQLite database. Second, we extract eight meaningful features per student using SQL queries. Third, a Random Forest model — trained on synthetic data — predicts the dropout probability. Fourth, this probability is converted to a risk score between 0 and 100. Finally, educators see these scores on a clean web dashboard with actionable recommendations. The entire system runs offline with no external dependencies or cloud services required."

---

## SLIDE 4: SYSTEM ARCHITECTURE

**Title:** System Architecture — 5 Layers

**Bullets** (describe each layer briefly):
- **Layer 1 — Data:** SQLite database, 4 tables, 25 students seeded
- **Layer 2 — Features:** 8 features extracted per student via SQL queries
- **Layer 3 — Synthetic Data:** 500 training rows generated with numpy distributions
- **Layer 4 — ML Model:** Random Forest, 100 trees, 80/20 train-test split
- **Layer 5 — Dashboard:** Streamlit, 4 screens, real-time predictions

**Layout:** Vertical stack of 5 coloured boxes with downward-pointing arrows connecting them. Each box is one layer, colour-coded:
- Layer 1: Dark blue (#1E3A5F)
- Layer 2: Blue (#3B82F6)
- Layer 3: Purple (#8B5CF6)
- Layer 4: Green (#10B981)
- Layer 5: Coral (#EF4444)

Place the architecture stack on the right half of the slide, with the bullets on the left half.

**Visual:** Layered architecture diagram with 5 stacked, coloured, rounded-corner boxes connected by arrows.

**Speaker Notes:**
"Let me show you the architecture. At the bottom is our SQLite database storing all student, course, enrollment, and assessment data across four normalised tables. Above that is the feature extraction layer — eight features computed per student using parameterised SQL queries. Because we have no historical dropout records to learn from, we generated 500 synthetic training rows using realistic statistical distributions with numpy. The Random Forest model was trained on this data with a balanced class weight to handle the 80-20 class imbalance. Finally, the Streamlit dashboard sits on top, consuming predictions and presenting them visually to educators."

---

## SLIDE 5: THE 8 FEATURES

**Title:** What We Measure — 8 Dropout Risk Features

**Layout:** Present as a 2-column table on the slide with alternating row colours (#F9FAFB / #FFFFFF). Add a small icon to each row.

**Left column (Features 1–4):**

| # | Feature               | Description                          |
|---|-----------------------|--------------------------------------|
| 1 | avg_marks             | Mean score across all assessments    |
| 2 | courses_enrolled      | Total courses registered             |
| 3 | courses_with_zero     | Courses with zero average marks      |
| 4 | assessments_missing   | Assessments not submitted            |

**Right column (Features 5–8):**

| # | Feature               | Description                          |
|---|-----------------------|--------------------------------------|
| 5 | grade_f_count         | Courses failed (below 40%)           |
| 6 | lowest_course_score   | Worst course performance             |
| 7 | marks_trend ⭐        | Slope: is performance improving?     |
| 8 | completion_rate ⭐    | Fraction of work submitted           |

**Highlight** features 1, 7, and 8 in a different colour (#DBEAFE blue background) with a note centred below the table:
> "⭐ Top 3 predictors identified by the trained model"

**Visual:** Feature table with colour highlights on the top 3 predictors. Use small icons (📊, 📚, ⚠️, 📝, ❌, 📉, 📈, ✅) next to each feature name.

**Speaker Notes:**
"We represent every student as a vector of eight numbers. These features cover three dimensions: marks quality — like average marks and grade failure count; submission behaviour — like missing assessments and completion rate; and trend — the marks trend slope tells us whether a student is improving or declining over time. The model identifies which of these features matter most during training. In our results, average marks, completion rate, and marks trend emerged as the top three predictors — which aligns completely with the educational data mining research literature."

---

## SLIDE 6: THE AI MODEL

**Title:** The Machine Learning Model

**Bullets:**
- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Training data:** 500 synthetic rows, ~20% dropout rate
- **Split:** 80% train (400 rows), 20% test (100 rows)
- **Configuration:** 100 trees, max_depth=8, class_weight='balanced'
- **Handles class imbalance** automatically via balanced weights
- **Outputs:** dropout probability → multiplied by 100 = risk score

**Code block** (dark background #1F2937, monospace font, on right side of slide):
```python
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
```

**Layout:** Left side — bullets (60% width). Right side — code block screenshot + a small hand-drawn-style decision tree diagram below the code (3 levels deep, showing how a single tree splits on avg_marks and completion_rate).

**Visual:** Simple decision tree sketch + code block screenshot. Tree should show: root node splits on avg_marks < 0.45, left child splits on completion_rate < 0.60, leading to "Dropout" leaf.

**Speaker Notes:**
"For our machine learning model, we chose the Random Forest Classifier from scikit-learn. Random Forest builds multiple decision trees and aggregates their votes — this reduces overfitting and gives more stable predictions than a single tree. We configured it with 100 trees and a maximum depth of 8 to prevent overfitting on our 500-row dataset. Because our synthetic dataset has roughly 20 percent dropouts and 80 percent non-dropouts, we set class weight to balanced so the model does not simply predict no dropout for everyone. The model outputs a probability which we multiply by 100 to get the final risk score between 0 and 100."

---

## SLIDE 7: RESULTS

**Title:** Model Results & Performance

**Left side — Metrics table** (use a clean table with blue header row):

| Metric    | Score     |
|-----------|-----------|
| Accuracy  | [ACC]%    |
| Precision | [PREC]%   |
| Recall    | [REC]%    |
| F1 Score  | [F1]%     |

**Below the metrics table, add 3 insight bullets:**
- ✅ High Recall = most at-risk students are correctly identified
- ✅ Precision trade-off is acceptable — false alarms are low cost
- ✅ avg_marks and completion_rate are the strongest predictors

**Right side** — Feature importance chart:
Insert `model/feature_importance.png` here. Scale to fill the right 50% of the slide.

**Layout:** 2 columns. Left: metrics table + insight bullets (50%). Right: feature importance chart image (50%).

**Visual:** Feature importance PNG from `model/feature_importance.png`.

**Speaker Notes:**
"Our model achieved [ACC] percent accuracy on the held-out test set. More importantly, the recall score is [REC] percent — this means we correctly identify [REC] percent of all students who are truly at risk. In this domain, recall matters more than precision because a missed dropout is far more costly than a false alarm. A false alarm just means a teacher has an extra check-in with a student who is actually fine — that is not a problem. On the right you can see the feature importance chart — average marks and completion rate are by far the strongest predictors, which aligns with the educational data mining literature. The marks trend slope is third, confirming that trajectory matters, not just current snapshot."

---

## SLIDE 8: DASHBOARD SCREENSHOTS

**Title:** The Streamlit Dashboard — 4 Screens

**Layout:** 2×2 grid of screenshots with thin grey borders and drop shadows. Each screenshot takes roughly 45% of the slide width and 40% of the height.

| Position     | Screen                              | Caption                              |
|-------------|-------------------------------------|--------------------------------------|
| Top-left    | Screen 1 — Overview Dashboard       | "Risk table with 25 students"        |
| Top-right   | Screen 2 — Student Detail           | "Gauge + top risk factors"           |
| Bottom-left | Screen 3 — Model Insights           | "Accuracy metrics + confusion matrix"|
| Bottom-right| Screen 4 — Simulate Student         | "Live risk simulation"               |

Caption text should be 10pt, grey (#6B7280), centred below each screenshot.

**Visual:** 4 browser screenshots arranged in a 2×2 grid. Take these by running `streamlit run app.py` and capturing each screen from the browser.

**Speaker Notes:**
"Here are the four screens of the dashboard. The Overview screen shows all 25 students in a colour-coded risk table — red for high, amber for medium, green for low — with metric cards at the top showing summary counts. The Student Detail screen shows a semicircle gauge chart with the student's exact score, their top three risk factors, and a recommended action. The Model Insights screen shows all accuracy metrics, the feature importance chart, and a confusion matrix heatmap for model transparency. Finally, the Simulate screen lets anyone type in hypothetical marks and instantly see a live prediction — which I will demonstrate now."

---

## SLIDE 9: LIVE DEMO

**Title:** Live Demonstration

**Content** (not bullets — large numbered steps centred on a dark background #1E293B with white text):

🎯 **Watch live:**

1. **Overview Dashboard** — 25 students, coloured risk scores
2. **High Risk Student** — gauge chart, 3 factors, recommendation
3. **Model Insights** — accuracy, feature importance, confusion matrix
4. **Simulate Screen** — type bad marks, watch score jump to High

**Layout:** Dark background (#1E293B), large numbered steps in 24pt white font, generous spacing. Minimal text — let the demo speak for itself. A "▶ LIVE DEMO" banner in red (#EF4444) at the top of the slide.

**Visual:** A "LIVE DEMO" banner or a simple play button icon. No charts on this slide.

**Speaker Notes:**
"I will now demonstrate the live system. I will switch to the browser where the Streamlit dashboard is already running. First I will show the Overview screen with all 25 students ranked by risk score. Then I will click into a High risk student to show their detail view with the gauge chart and risk factors. Then I will open the Model Insights tab to show the AI metrics and confusion matrix. Finally I will go to the Simulate screen, enter some very poor marks, and you will see the risk score jump to High in real time. Let me switch to the browser now."

---

## SLIDE 10: CONCLUSION & FUTURE SCOPE

**Title:** Conclusion & Future Scope

**Left column — What we achieved** (use green tick emojis, 16pt):
- ✅ Structured SQLite database with 4 normalised tables
- ✅ 8 features extracted per student via SQL queries
- ✅ Random Forest trained — [ACC]% accuracy
- ✅ Live Streamlit dashboard with 4 interactive screens
- ✅ Rule-based fallback for graceful degradation
- ✅ Fully offline — no external APIs or cloud services

**Right column — Future scope** (use crystal ball emojis, 16pt):
- 🔮 Connect to real LMS (Moodle/Canvas) via API
- 🔮 Add SHAP explainability for deeper per-student insights
- 🔮 Auto-email alerts when student crosses High risk threshold
- 🔮 Retrain monthly as new semester data arrives
- 🔮 Extend to attendance and engagement features

**Bottom** — centred, 28pt, bold:
> "Thank you. Questions welcome."

**Layout:** Two equal-width columns with the achievements on the left and future scope on the right. A thin vertical divider line between columns. The thank-you line spans the full width at the bottom.

**Visual:** Two-column layout with tick and rocket emoji bullets. Clean white background with a subtle blue gradient at the top edge.

**Speaker Notes:**
"To conclude — we have built a complete AI-powered dropout risk prediction system from scratch in four phases. The system correctly identifies at-risk students with [ACC] percent accuracy, presents results in a clean educator-friendly dashboard, and gracefully falls back to a rule-based approach if the model file is unavailable. The main limitation is that we trained on synthetic data — in production, this would be retrained on real institutional records from the university's LMS. Future work includes connecting to a live Learning Management System like Moodle, adding SHAP explainability for deeper per-student analysis, and setting up automated email alerts so advisors are notified immediately when a student crosses into the High risk category. Thank you for your attention. I am happy to take any questions."
