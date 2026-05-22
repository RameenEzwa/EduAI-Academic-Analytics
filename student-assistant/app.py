# ============================================================
# AI Student Performance Assistant
# Supports SDG 4: Quality Education
# ============================================================
# This program loads a student dataset, analyzes performance,
# predicts exam scores using machine learning, and gives study
# recommendations — all through a simple text menu.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # Use non-interactive backend (works in Replit)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# GLOBAL VARIABLES
# ─────────────────────────────────────────────────────────────
DATASET_FILE = "StudentPerformanceFactors.csv"
WEAK_THRESHOLD = 65          # Scores below this are considered "weak"
OUTPUT_DIR = "charts"        # Folder where chart images are saved


# ─────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def print_divider() -> None:
    """Print a thin divider line."""
    print("-" * 55)


def ensure_output_dir() -> None:
    """Create the charts output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_chart(filename: str) -> None:
    """Save the current matplotlib figure and close it."""
    ensure_output_dir()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  Chart saved → {filepath}")


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset into a pandas DataFrame.
    Exits gracefully if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"\n[ERROR] Dataset file '{filepath}' not found.")
        print("Make sure 'StudentPerformanceFactors.csv' is in the same folder as app.py.")
        raise SystemExit(1)

    df = pd.read_csv(filepath)

    # Drop rows where the target column is missing
    df.dropna(subset=['Exam_Score'], inplace=True)

    # Fill remaining missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill remaining missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# ─────────────────────────────────────────────────────────────
# 2. DATASET SUMMARY
# ─────────────────────────────────────────────────────────────

def view_dataset_summary(df: pd.DataFrame) -> None:
    """Display a high-level overview of the dataset."""
    print_header("Dataset Summary")
    print(f"  Total students   : {len(df)}")
    print(f"  Total columns    : {len(df.columns)}")
    print(f"  Missing values   : {df.isnull().sum().sum()}")
    print_divider()
    print("  Column names:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2}. {col}")
    print_divider()
    print("\n  Numeric statistics (key columns):")
    key_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours',
                'Previous_Scores', 'Exam_Score']
    key_cols = [c for c in key_cols if c in df.columns]
    print(df[key_cols].describe().round(2).to_string())


# ─────────────────────────────────────────────────────────────
# 3. AVERAGE STUDENT SCORES
# ─────────────────────────────────────────────────────────────

def show_average_scores(df: pd.DataFrame) -> None:
    """Show average exam scores broken down by several categories."""
    print_header("Average Student Scores")

    overall_avg = df['Exam_Score'].mean()
    print(f"  Overall average exam score : {overall_avg:.2f}")
    print_divider()

    # Break down by gender if column exists
    if 'Gender' in df.columns:
        print("  Average score by Gender:")
        gender_avg = df.groupby('Gender')['Exam_Score'].mean().round(2)
        for g, score in gender_avg.items():
            print(f"    {g:<12}: {score}")
        print_divider()

    # Break down by school type
    if 'School_Type' in df.columns:
        print("  Average score by School Type:")
        school_avg = df.groupby('School_Type')['Exam_Score'].mean().round(2)
        for s, score in school_avg.items():
            print(f"    {s:<12}: {score}")
        print_divider()

    # Break down by parental involvement
    if 'Parental_Involvement' in df.columns:
        print("  Average score by Parental Involvement:")
        parent_avg = df.groupby('Parental_Involvement')['Exam_Score'].mean().round(2)
        for p, score in parent_avg.items():
            print(f"    {p:<12}: {score}")


# ─────────────────────────────────────────────────────────────
# 4. DETECT WEAK-PERFORMING STUDENTS
# ─────────────────────────────────────────────────────────────

def detect_weak_students(df: pd.DataFrame) -> None:
    """
    Identify students scoring below WEAK_THRESHOLD and
    show patterns among them.
    """
    print_header(f"Weak-Performing Students (score < {WEAK_THRESHOLD})")

    weak_df = df[df['Exam_Score'] < WEAK_THRESHOLD].copy()
    print(f"  Number of weak students   : {len(weak_df)}")
    print(f"  Percentage of total       : {100 * len(weak_df) / len(df):.1f}%")
    print(f"  Their average score       : {weak_df['Exam_Score'].mean():.2f}")
    print_divider()

    if len(weak_df) == 0:
        print("  No weak-performing students found.")
        return

    print("  Common traits of weak students:")

    if 'Hours_Studied' in df.columns:
        avg_hrs_weak = weak_df['Hours_Studied'].mean()
        avg_hrs_all  = df['Hours_Studied'].mean()
        print(f"    Avg hours studied : {avg_hrs_weak:.1f}  (all students: {avg_hrs_all:.1f})")

    if 'Attendance' in df.columns:
        avg_att_weak = weak_df['Attendance'].mean()
        avg_att_all  = df['Attendance'].mean()
        print(f"    Avg attendance    : {avg_att_weak:.1f}%  (all students: {avg_att_all:.1f}%)")

    if 'Sleep_Hours' in df.columns:
        avg_slp_weak = weak_df['Sleep_Hours'].mean()
        avg_slp_all  = df['Sleep_Hours'].mean()
        print(f"    Avg sleep hours   : {avg_slp_weak:.1f}  (all students: {avg_slp_all:.1f})")

    if 'Motivation_Level' in df.columns:
        print(f"    Top motivation level: {weak_df['Motivation_Level'].mode()[0]}")

    print_divider()
    print("  Sample of first 10 weak students:")
    sample_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Exam_Score']
    sample_cols = [c for c in sample_cols if c in df.columns]
    print(weak_df[sample_cols].head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 5. PREDICT PERFORMANCE BASED ON STUDY HOURS (Linear Regression)
# ─────────────────────────────────────────────────────────────

def predict_performance(df: pd.DataFrame) -> None:
    """
    Train a simple linear regression model using Hours_Studied
    to predict Exam_Score, then let the user enter study hours
    for a personalised prediction.
    """
    print_header("Predict Performance by Study Hours")

    if 'Hours_Studied' not in df.columns:
        print("  'Hours_Studied' column not found in dataset.")
        return

    # Prepare features and target
    X = df[['Hours_Studied']].values
    y = df['Exam_Score'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  Model trained  : Linear Regression")
    print(f"  RMSE on test   : {rmse:.2f}  (lower = better)")
    print_divider()

    # Interactive prediction
    while True:
        try:
            hours_input = input("  Enter study hours per week to predict score (or 'back'): ").strip()
            if hours_input.lower() == 'back':
                return
            hours = float(hours_input)
            if hours < 0 or hours > 168:
                print("  [!] Please enter a value between 0 and 168 hours.")
                continue
            predicted_score = model.predict([[hours]])[0]
            predicted_score = max(0, min(100, predicted_score))   # Clamp to valid range
            print(f"\n  Predicted Exam Score for {hours:.1f} study hours/week: {predicted_score:.1f}")
            break
        except ValueError:
            print("  [!] Invalid input. Please enter a number.")


# ─────────────────────────────────────────────────────────────
# 6. STUDY RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────

def give_recommendations(df: pd.DataFrame) -> None:
    """
    Ask the user for their current study habits and return
    personalised study recommendations.
    """
    print_header("Personalised Study Recommendations")
    print("  Answer a few questions to get tailored advice.\n")

    def get_number(prompt: str, min_val: float, max_val: float) -> float:
        """Helper: prompt the user until a valid number in range is given."""
        while True:
            try:
                val = float(input(f"  {prompt}: ").strip())
                if min_val <= val <= max_val:
                    return val
                print(f"  [!] Enter a value between {min_val} and {max_val}.")
            except ValueError:
                print("  [!] Please enter a valid number.")

    hours     = get_number("How many hours do you study per week?", 0, 168)
    sleep     = get_number("How many hours do you sleep per night?", 0, 24)
    attendance = get_number("What is your attendance percentage?", 0, 100)

    print_divider()
    print("  Your Recommendations:")
    print()

    # Dataset averages for comparison
    avg_hours = df['Hours_Studied'].mean() if 'Hours_Studied' in df.columns else 20
    avg_sleep = df['Sleep_Hours'].mean()   if 'Sleep_Hours'   in df.columns else 7
    avg_att   = df['Attendance'].mean()    if 'Attendance'    in df.columns else 85

    # Study hours advice
    if hours < avg_hours * 0.7:
        print(f"  [Study Hours] You study {hours:.0f} hrs/week — below average ({avg_hours:.0f} hrs).")
        print("   Tip: Aim for at least 20 hours/week. Break it into 2-3 hour sessions.")
    elif hours > avg_hours * 1.5:
        print(f"  [Study Hours] You study {hours:.0f} hrs/week — above average! Great effort.")
        print("   Tip: Make sure to take breaks to avoid burnout (Pomodoro technique works well).")
    else:
        print(f"  [Study Hours] Good — {hours:.0f} hrs/week is close to the class average ({avg_hours:.0f} hrs).")
        print("   Tip: Focus on quality over quantity — use active recall and spaced repetition.")

    print()

    # Sleep advice
    if sleep < 6:
        print(f"  [Sleep] {sleep:.0f} hrs/night is too low. Sleep deprivation hurts memory consolidation.")
        print("   Tip: Target 7–9 hours. Even one extra hour can improve test performance.")
    elif sleep > 10:
        print(f"  [Sleep] {sleep:.0f} hrs/night is quite high. Oversleeping can reduce alertness.")
        print("   Tip: A consistent 7–8 hour schedule is optimal for learning.")
    else:
        print(f"  [Sleep] {sleep:.0f} hrs/night is healthy. Keep this routine!")

    print()

    # Attendance advice
    if attendance < 75:
        print(f"  [Attendance] {attendance:.0f}% attendance is very low — you may be missing key content.")
        print("   Tip: Attend at least 80% of classes. Review missed notes within 24 hours.")
    elif attendance < avg_att:
        print(f"  [Attendance] {attendance:.0f}% is below the class average ({avg_att:.0f}%).")
        print("   Tip: Every class matters. Consistent attendance correlates with higher scores.")
    else:
        print(f"  [Attendance] {attendance:.0f}% — excellent attendance! Keep it up.")

    print()
    print("  General Tips:")
    print("   • Use the Pomodoro technique (25 min study / 5 min break)")
    print("   • Practice retrieval: test yourself instead of re-reading")
    print("   • Study with peers — explaining concepts cements understanding")
    print("   • Limit screen time 1 hour before bed to improve sleep quality")
    print("   • Seek help from teachers or tutors when stuck — don't wait")


# ─────────────────────────────────────────────────────────────
# 7. SHOW GRAPHS / CHARTS
# ─────────────────────────────────────────────────────────────

def show_graphs(df: pd.DataFrame) -> None:
    """Display a sub-menu for choosing which chart to generate."""

    chart_menu = {
        '1': ('Exam Score Distribution',    _chart_score_distribution),
        '2': ('Study Hours vs Exam Score',   _chart_hours_vs_score),
        '3': ('Attendance vs Exam Score',    _chart_attendance_vs_score),
        '4': ('Average Score by Gender',     _chart_gender_avg),
        '5': ('Score by Motivation Level',   _chart_motivation),
        '6': ('All Charts at Once',          None),
    }

    while True:
        print_header("Charts & Graphs")
        for key, (label, _) in chart_menu.items():
            print(f"  [{key}] {label}")
        print("  [0] Back to Main Menu")
        print_divider()

        choice = input("  Select a chart: ").strip()

        if choice == '0':
            return
        elif choice == '6':
            print("\n  Generating all charts…")
            for key, (label, func) in chart_menu.items():
                if func is not None:
                    func(df)
            print("  All charts saved.")
        elif choice in chart_menu:
            label, func = chart_menu[choice]
            if func:
                func(df)
        else:
            print("  [!] Invalid choice. Please enter a number from the menu.")


def _chart_score_distribution(df: pd.DataFrame) -> None:
    """Bar chart: distribution of exam scores in bins."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['Exam_Score'], bins=20, color='steelblue', edgecolor='white')
    ax.set_title('Exam Score Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Exam Score')
    ax.set_ylabel('Number of Students')
    ax.axvline(df['Exam_Score'].mean(), color='red', linestyle='--',
               label=f"Mean: {df['Exam_Score'].mean():.1f}")
    ax.legend()
    plt.tight_layout()
    save_chart('score_distribution.png')


def _chart_hours_vs_score(df: pd.DataFrame) -> None:
    """Scatter plot: Hours Studied vs Exam Score with regression line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['Hours_Studied'], df['Exam_Score'],
               alpha=0.4, color='royalblue', s=15)

    # Add trend line
    m, b = np.polyfit(df['Hours_Studied'], df['Exam_Score'], 1)
    x_line = np.linspace(df['Hours_Studied'].min(), df['Hours_Studied'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='red', linewidth=2, label='Trend line')

    ax.set_title('Study Hours vs Exam Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hours Studied per Week')
    ax.set_ylabel('Exam Score')
    ax.legend()
    plt.tight_layout()
    save_chart('hours_vs_score.png')


def _chart_attendance_vs_score(df: pd.DataFrame) -> None:
    """Scatter plot: Attendance vs Exam Score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['Attendance'], df['Exam_Score'],
               alpha=0.4, color='mediumseagreen', s=15)
    ax.set_title('Attendance vs Exam Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Attendance (%)')
    ax.set_ylabel('Exam Score')
    plt.tight_layout()
    save_chart('attendance_vs_score.png')


def _chart_gender_avg(df: pd.DataFrame) -> None:
    """Bar chart: average exam score by gender."""
    if 'Gender' not in df.columns:
        print("  'Gender' column not found.")
        return
    gender_avg = df.groupby('Gender')['Exam_Score'].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(gender_avg.index, gender_avg.values,
                  color=['#4c72b0', '#dd8452'], edgecolor='white')
    ax.bar_label(bars, fmt='%.1f', padding=3)
    ax.set_title('Average Exam Score by Gender', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Exam Score')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    save_chart('gender_avg_score.png')


def _chart_motivation(df: pd.DataFrame) -> None:
    """Bar chart: average exam score by motivation level."""
    if 'Motivation_Level' not in df.columns:
        print("  'Motivation_Level' column not found.")
        return
    mot_avg = df.groupby('Motivation_Level')['Exam_Score'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(mot_avg.index, mot_avg.values, color='mediumpurple', edgecolor='white')
    ax.bar_label(bars, fmt='%.1f', padding=3)
    ax.set_title('Avg Score by Motivation Level', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Exam Score')
    ax.set_xlim(0, 100)
    plt.tight_layout()
    save_chart('motivation_avg_score.png')


# ─────────────────────────────────────────────────────────────
# 8. MACHINE LEARNING — FULL MODEL
# ─────────────────────────────────────────────────────────────

def run_ml_model(df: pd.DataFrame) -> None:
    """
    Train two machine learning models:
      (a) Linear Regression — predict exact Exam_Score
      (b) Random Forest Classifier — predict performance category
          (Weak / Average / Strong)
    Shows accuracy metrics and lets the user predict their own score.
    """
    print_header("Machine Learning Model")

    # ── Encode categorical columns ────────────────────────────
    df_ml = df.copy()
    encoders = {}
    cat_cols = df_ml.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le

    # ── Features and targets ──────────────────────────────────
    feature_cols = [c for c in df_ml.columns if c != 'Exam_Score']
    X = df_ml[feature_cols].values
    y_score = df_ml['Exam_Score'].values

    # Create performance category labels
    def categorise(score):
        if score < 60:
            return 'Weak'
        elif score < 75:
            return 'Average'
        else:
            return 'Strong'

    y_category = np.array([categorise(s) for s in y_score])

    # Train / test split
    X_train, X_test, ys_train, ys_test, yc_train, yc_test = train_test_split(
        X, y_score, y_category, test_size=0.2, random_state=42)

    # ── (a) Linear Regression ─────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, ys_train)
    ys_pred = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(ys_test, ys_pred))

    print("  (a) Linear Regression — Predict Exact Score")
    print(f"      RMSE : {rmse:.2f}  (avg prediction error in score points)")
    print_divider()

    # ── (b) Random Forest Classifier ─────────────────────────
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, yc_train)
    yc_pred  = rf.predict(X_test)
    accuracy = accuracy_score(yc_test, yc_pred)

    print("  (b) Random Forest — Predict Category (Weak/Average/Strong)")
    print(f"      Accuracy : {accuracy * 100:.1f}%")
    print_divider()

    # Top 5 features
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top5 = importances.sort_values(ascending=False).head(5)
    print("  Top 5 most influential features:")
    for feat, imp in top5.items():
        print(f"    {feat:<30} {imp:.4f}")

    print_divider()
    print("  Prediction mode: enter feature values to predict your score.\n")
    print("  Using the two most important numeric features for quick input:")
    print("    1. Hours_Studied  2. Attendance\n")

    while True:
        try:
            hrs_input = input("  Hours studied per week (or 'back'): ").strip()
            if hrs_input.lower() == 'back':
                return
            hrs = float(hrs_input)

            att_input = input("  Attendance percentage (0-100): ").strip()
            att = float(att_input)

            # Build a row filled with median values, then override knowns
            median_row = df_ml[feature_cols].median().values.copy()
            if 'Hours_Studied' in feature_cols:
                median_row[feature_cols.index('Hours_Studied')] = hrs
            if 'Attendance' in feature_cols:
                median_row[feature_cols.index('Attendance')] = att

            predicted_score    = lr.predict([median_row])[0]
            predicted_category = rf.predict([median_row])[0]
            predicted_score    = max(0, min(100, predicted_score))

            print(f"\n  ── Prediction Results ──────────────────────────")
            print(f"  Predicted Exam Score    : {predicted_score:.1f}")
            print(f"  Performance Category    : {predicted_category}")
            print(f"  ────────────────────────────────────────────────\n")
            break

        except ValueError:
            print("  [!] Invalid input — please enter numeric values.")


# ─────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────

def print_main_menu() -> None:
    """Print the main application menu."""
    print_header("AI Student Performance Assistant")
    print("  Supports SDG 4: Quality Education")
    print_divider()
    print("  [1] View Dataset Summary")
    print("  [2] Show Average Student Scores")
    print("  [3] Detect Weak-Performing Students")
    print("  [4] Predict Performance by Study Hours")
    print("  [5] Get Study Recommendations")
    print("  [6] Show Graphs / Charts")
    print("  [7] Run Full ML Model")
    print("  [0] Exit")
    print_divider()


def main() -> None:
    """
    Entry point: load data then run the interactive menu loop.
    All invalid inputs are caught and the menu is shown again.
    """
    print("\n  Loading dataset…")
    df = load_dataset(DATASET_FILE)
    print(f"  Dataset loaded: {len(df)} student records, {len(df.columns)} features.")

    menu_actions = {
        '1': lambda: view_dataset_summary(df),
        '2': lambda: show_average_scores(df),
        '3': lambda: detect_weak_students(df),
        '4': lambda: predict_performance(df),
        '5': lambda: give_recommendations(df),
        '6': lambda: show_graphs(df),
        '7': lambda: run_ml_model(df),
    }

    while True:
        print_main_menu()
        choice = input("  Enter your choice: ").strip()

        if choice == '0':
            print("\n  Thank you for using the AI Student Performance Assistant.")
            print("  Keep studying hard — SDG 4 Quality Education starts with you!\n")
            break
        elif choice in menu_actions:
            try:
                menu_actions[choice]()
            except KeyboardInterrupt:
                print("\n  (Interrupted — returning to main menu)")
            except Exception as e:
                print(f"\n  [ERROR] Something went wrong: {e}")
                print("  Returning to main menu…")
        else:
            print("\n  [!] Invalid choice. Please enter a number from 0 to 7.")

        input("\n  Press Enter to continue…")


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
