# Student Dropout Risk Predictor — Live Demo Script
## Exact Words to Say, Clicks to Make, and Fallback Plans

---

## PRE-DEMO CHECKLIST (Do This 10 Minutes Before Presenting)

- [ ] Open a terminal in the `dropout_predictor/` folder
- [ ] Run `streamlit run app.py` — confirm it opens in browser at http://localhost:8501
- [ ] Confirm all 25 students appear on the Overview screen in the risk table
- [ ] Confirm at least one High risk student exists in the table (look for red "High" badges)
- [ ] Note the name of the highest-risk student — you will click this name live (e.g., "Tanvi Saxena")
- [ ] Click "Model Insights" in the sidebar — confirm accuracy metrics appear (model/metrics.pkl must exist)
- [ ] Click "Simulate Student" in the sidebar — set avg_marks to 0.10 and click Predict Risk — confirm a High result appears
- [ ] Zoom browser to 90% so everything fits on screen without scrolling (`Ctrl + -` on Windows)
- [ ] Close all other browser tabs — only the Streamlit tab should be open
- [ ] Set screen resolution to 1920×1080 if projecting
- [ ] Turn off all notifications on your laptop (Windows: Focus Assist → Alarms Only)
- [ ] Keep the terminal window visible but minimised — you may need it for fallback recovery

---

## DEMO STEP 1: OVERVIEW DASHBOARD (60 seconds)

**ACTION:** Switch from the PowerPoint to the browser. Screen 1 — Overview Dashboard should already be visible. If not, click "📊 Overview Dashboard" in the sidebar.

**DIALOGUE:**
"This is the Overview Dashboard. At the top you can see four metric cards — total students, and the count in each risk category: High, Medium, and Low. Below is the full student risk table. Each student has a coloured category badge — red for High and green for Low. The table is sorted by risk score, so the most at-risk students appear at the very top. This gives a teacher an instant, at-a-glance view of who needs attention today."

**ACTION:** Point to the donut chart on the right side of the screen.

**DIALOGUE:**
"On the right is the risk distribution chart — a donut showing the proportion of students in each category. You can see that a significant number are flagged as High risk — these are the students who need immediate intervention."

**ACTION:** Point to the highest-risk student's row in the table (first row).

**DIALOGUE:**
"Notice this student at the top — they have the highest risk score. Let me now show you their full detail view."

**ACTION:** Scroll down to the student selector at the bottom of the Overview page. Select the highest-risk student name from the dropdown. Click "View Student Detail →".

---

## DEMO STEP 2: STUDENT DETAIL (90 seconds)

**ACTION:** The page should now show Screen 2 — Student Detail with the selected student. If it navigated to the Student Detail tab instead, select the same student from the dropdown at the top.

**DIALOGUE:**
"This is the Student Detail screen. On the left you see a gauge chart — like a speedometer — showing this student's risk score out of 100. The needle is deep into the red zone, indicating High risk."

**ACTION:** Point to the "High" badge below the gauge.

**DIALOGUE:**
"Below the gauge is the category badge — clearly marked as High."

**ACTION:** Point to the Top Risk Factors section on the right.

**DIALOGUE:**
"On the right, the model has identified the three biggest risk factors for this specific student. These are not generic warnings — they are computed from this student's actual database records. For example, you might see that they have very low average marks, missing assessments, and zero marks in multiple courses. Each factor is ranked by the model's feature importance scores."

**ACTION:** Point to the Recommended Action box.

**DIALOGUE:**
"Below the risk factors is a recommended action — in this case, immediate academic counselling is advised, with a suggestion to contact the student and their guardian within 48 hours. A teacher using this system knows exactly what to do next — no interpretation needed."

**ACTION:** Scroll down to the Student Feature Profile bar chart.

**DIALOGUE:**
"The bar chart below shows all eight features for this student. Red bars indicate problem areas — you can see that completion rate and average marks are critically low. Blue bars indicate features that are within normal range. This chart gives the complete picture of why the model flagged this student."

---

## DEMO STEP 3: MODEL INSIGHTS (60 seconds)

**ACTION:** Click "🤖 Model Insights" in the sidebar.

**DIALOGUE:**
"This is the Model Insights screen — the AI transparency layer. At the top you can see the model's performance metrics. We achieved strong accuracy, and more importantly, high recall. Recall is especially important here — it tells us what fraction of actual at-risk students the model correctly identified. A high recall means we are not missing students who genuinely need help."

**ACTION:** Point to the feature importance chart.

**DIALOGUE:**
"This chart shows what the Random Forest model considers most important when making its predictions. Average marks and completion rate are at the top — which matches exactly what the educational research literature says about the strongest predictors of student dropout."

**ACTION:** Scroll down to the confusion matrix heatmap.

**DIALOGUE:**
"The confusion matrix shows how the model performed on the 100-row test set. The diagonal cells — top-left and bottom-right — show correct predictions. The off-diagonal cells show errors. You can see the model makes very few false negatives — that is, it rarely misses a student who is actually at risk. This is critical because a missed at-risk student could drop out without any intervention."

---

## DEMO STEP 4: SIMULATE STUDENT (90 seconds)

**ACTION:** Click "🧪 Simulate Student" in the sidebar.

**DIALOGUE:**
"This is my favourite screen. It lets anyone — an evaluator, a teacher, even a student — type in hypothetical marks and instantly see the predicted risk score. Watch what happens when I enter the values for a struggling student."

**ACTION:** Set these values live on screen, slowly so the audience can see each change:
- avg_marks → slide left to **0.20**
- courses_enrolled → leave at 4
- courses_with_zero_marks → type in **3**
- assessments_missing → type in **8**
- grade_f_count → type in **4**
- lowest_course_score → slide left to **0.10**
- marks_trend → slide left to **-0.50**
- completion_rate → slide left to **0.25**

**ACTION:** Click the "🔮 Predict Risk" button.

**DIALOGUE:**
"The gauge immediately shows a High risk score — well above 70 out of 100. The system has identified three risk factors and recommended immediate academic counselling. All of this happens in under one second, entirely offline, with no external services or internet connection required."

**ACTION:** Pause for 3 seconds to let the audience absorb the result.

**DIALOGUE:**
"Now let me show you the opposite case — an excellent student."

**ACTION:** Set all sliders to the best case:
- avg_marks → slide right to **0.90**
- courses_with_zero_marks → type in **0**
- assessments_missing → type in **0**
- grade_f_count → type in **0**
- lowest_course_score → slide right to **0.75**
- marks_trend → slide right to **0.30**
- completion_rate → slide right to **0.95**

**ACTION:** Click "🔮 Predict Risk" again.

**DIALOGUE:**
"Now you can see the score drops to Low — the system correctly identifies them as a healthy, on-track student with no risk factors detected. The recommended action changes to routine monitoring and encouragement. This live simulation shows that the predictions are intuitive, responsive, and align with what any educator would expect."

**ACTION:** Switch back to the PowerPoint (Slide 10 — Conclusion).

---

## FALLBACK PLANS (If Something Goes Wrong Live)

### PROBLEM: App won't start (streamlit not found)

**FIX:** Open a terminal and run:
```
pip install streamlit
streamlit run app.py
```
If still failing, show the terminal-based output instead:
```
python -c "import sqlite3; from model.predict import predict_all_students; conn=sqlite3.connect('dropout_predictor.db'); [print(r['student_name'], r['risk_score'], r['risk_category']) for r in predict_all_students(conn)]"
```
This prints all 25 students with risk scores as text output.

---

### PROBLEM: Screen shows "Model not trained yet"

**FIX:** Open a second terminal tab and run:
```
python model/train.py
```
Wait 10 seconds for training to complete, then refresh the browser with F5. The metrics will appear.

---

### PROBLEM: Database not found error

**FIX:** Run from the `dropout_predictor/` folder:
```
python db/seed.py
```
This recreates the database in under 2 seconds. Refresh the browser.

---

### PROBLEM: Feature importance PNG not showing

**FIX:** Open the file directly in any image viewer:
```
start model\feature_importance.png
```
Share your screen showing the image viewer instead. Continue the demo — the rest of the dashboard does not depend on this file.

---

### PROBLEM: Browser crashes or freezes

**FIX:** In the terminal, press Ctrl+C to stop Streamlit, then immediately restart:
```
streamlit run app.py
```
Streamlit restarts in under 5 seconds. Say to the audience: "Let me just restart the dashboard — this takes a moment." Keep the terminal visible during the demo so you can restart quickly.

---

### PROBLEM: Evaluator asks a question you don't know

**Answer templates:**

**"Why not use a neural network?"**
> "Neural networks need much more data — typically thousands of examples — and are harder to explain. Random Forest gives us feature importances which tell teachers exactly WHY a student is flagged. In an educational context, that explainability is essential. A teacher needs to know what to do with the information, not just see a number."

**"Is the data real?"**
> "The training data is synthetic — generated to simulate realistic student patterns using numpy statistical distributions. The 25 students in the demo database were also seeded artificially with four different performance tiers. In production, this would be replaced with real institutional data from the university's Learning Management System."

**"How accurate is it really?"**
> "On our synthetic test set, the model achieves very high accuracy. We acknowledge this is an optimistic figure because training and test data come from the same synthetic distribution. Real-world accuracy would need to be validated against actual historical dropout records from the institution. That said, the feature importance rankings — which show average marks and completion rate as top predictors — align with published research on real-world data, so we are confident the model captures the right patterns."

**"Can it work with our university's data?"**
> "Yes. The system is designed to be modular. You would need to replace the SQLite database with a connection to your LMS, map your data to the same eight features, and retrain the model. The training script takes about 10 seconds to run, so retraining on new data each semester is very practical."

**"What about privacy?"**
> "The entire system runs offline on a local machine — no data is sent to any external server or cloud service. All processing happens locally, so the student data never leaves the institution's network."
