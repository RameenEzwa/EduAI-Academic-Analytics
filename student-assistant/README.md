# AI Student Performance Assistant

> An AI-powered, menu-driven Python application that analyses student performance data, trains machine learning models, and provides personalised study recommendations.

---

## SDG 4 — Quality Education

This project directly supports **UN Sustainable Development Goal 4: Quality Education** — *"Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all."*

By leveraging data and machine learning, the assistant helps educators and students:

- Identify at-risk learners early so targeted support can be provided.
- Understand which factors (study hours, attendance, motivation) most affect outcomes.
- Receive personalised study recommendations to improve performance.
- Visualise trends to guide evidence-based teaching decisions.

---

## Dataset

**Source:** [Kaggle — Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

**File:** `StudentPerformanceFactors.csv`

**Records:** ~6,607 students | **Features:** 20 columns

Key columns used:

| Column | Description |
|---|---|
| `Hours_Studied` | Weekly study hours |
| `Attendance` | Class attendance percentage |
| `Sleep_Hours` | Average nightly sleep |
| `Previous_Scores` | Prior academic scores |
| `Motivation_Level` | Low / Medium / High |
| `Exam_Score` | Final exam score (target variable) |

---

## Features

| # | Menu Option | Description |
|---|---|---|
| 1 | **Dataset Summary** | Rows, columns, missing values, key statistics |
| 2 | **Average Scores** | Breakdown by gender, school type, parental involvement |
| 3 | **Weak Students** | Flags students scoring below 65 and shows common traits |
| 4 | **Predict by Study Hours** | Linear regression — enter study hours to get a predicted score |
| 5 | **Study Recommendations** | Personalised tips based on your hours, sleep, and attendance |
| 6 | **Charts** | 5 matplotlib charts saved as PNG files in `charts/` |
| 7 | **Full ML Model** | Linear Regression + Random Forest; predict score & category |

### Charts generated

- `score_distribution.png` — histogram of exam scores
- `hours_vs_score.png` — scatter plot of study hours vs score
- `attendance_vs_score.png` — scatter plot of attendance vs score
- `gender_avg_score.png` — bar chart of average score by gender
- `motivation_avg_score.png` — bar chart of average score by motivation level

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.2.2 | Data loading, cleaning, analysis |
| `numpy` | 1.26.4 | Numerical computation |
| `matplotlib` | 3.8.4 | Charts and visualisations |
| `scikit-learn` | 1.4.2 | Linear Regression & Random Forest models |

---

## Project Structure

```
student-assistant/
├── app.py                        # Main application (all logic + menu)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── StudentPerformanceFactors.csv # Dataset (place here before running)
└── charts/                       # Auto-created — chart PNG outputs
```

---

## How to Run

### On Replit

1. Open the **Shell** tab.
2. Navigate into the project folder:
   ```bash
   cd student-assistant
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Use the numbered menu to explore features. Charts are saved to `charts/`.

### Locally (macOS / Linux / Windows)

```bash
# 1. Clone or download this project
git clone <your-repo-url>
cd student-assistant

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure the CSV file is in the same folder as app.py
#    Download from: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

# 5. Run
python app.py
```

---

## Sample Interaction

```
=======================================================
  AI Student Performance Assistant
=======================================================
  Supports SDG 4: Quality Education
-------------------------------------------------------
  [1] View Dataset Summary
  [2] Show Average Student Scores
  [3] Detect Weak-Performing Students
  [4] Predict Performance by Study Hours
  [5] Get Study Recommendations
  [6] Show Graphs / Charts
  [7] Run Full ML Model
  [0] Exit
-------------------------------------------------------
  Enter your choice: 4

  Enter study hours per week to predict score (or 'back'): 25
  Predicted Exam Score for 25.0 study hours/week: 71.4
```

---

## Machine Learning Models

### 1. Linear Regression (Menu 4 & 7)
- **Input:** Hours studied per week (+ all features in option 7)
- **Output:** Predicted numeric exam score
- **Metric:** RMSE (Root Mean Squared Error)

### 2. Random Forest Classifier (Menu 7)
- **Input:** All 19 student features
- **Output:** Performance category — `Weak` (<60), `Average` (60–74), `Strong` (≥75)
- **Metric:** Classification accuracy
- **Bonus:** Shows the top 5 most influential features

---

## Author Notes

- All input is validated — the program handles invalid entries gracefully.
- Charts are saved to the `charts/` folder (auto-created) as PNG files.
- Matplotlib uses the `Agg` (non-interactive) backend so it works in Replit without a display.
- The code is fully commented and organised into clear functions for readability.

---

*Built with Python · pandas · scikit-learn · matplotlib · Supporting SDG 4 Quality Education*
