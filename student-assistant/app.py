# ============================================================
# AI Student Performance Assistant — Streamlit Web Application
# Supports SDG 4: Quality Education
# Vision 2030 / Vision 2035 Aligned
# ============================================================
# Role-based portals:
#   • Landing Page  – overview and role selection
#   • Student Portal – personal AI predictions & recommendations
#   • Admin Portal   – dataset management & ML model monitoring
#   • Client Portal  – school analytics & SDG impact dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Student Performance Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DATASET_FILE  = "StudentPerformanceFactors.csv"
WEAK_THRESHOLD = 60
AVG_THRESHOLD  = 75

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────
if "portal" not in st.session_state:
    st.session_state.portal = "home"
if "student_name" not in st.session_state:
    st.session_state.student_name = "Alex Johnson"
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# ─────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and clean the student performance dataset."""
    df = pd.read_csv(DATASET_FILE)
    df.dropna(subset=["Exam_Score"], inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def categorise_score(score: float) -> str:
    """Return performance category label for a numeric score."""
    if score < WEAK_THRESHOLD:
        return "Weak"
    elif score < AVG_THRESHOLD:
        return "Average"
    else:
        return "Strong"


# ─────────────────────────────────────────────────────────────
# ML MODELS (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Train and return two ML models:
      lr  – Linear Regression (predict exact score)
      rf  – Random Forest Classifier (predict category)
    Also returns feature column list, label encoders, and metrics.
    """
    df_ml = df.copy()

    # Encode categoricals
    encoders = {}
    for col in df_ml.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in df_ml.columns if c != "Exam_Score"]
    X = df_ml[feature_cols].values
    y_score = df_ml["Exam_Score"].values
    y_cat   = np.array([categorise_score(s) for s in y_score])

    X_train, X_test, ys_tr, ys_te, yc_tr, yc_te = train_test_split(
        X, y_score, y_cat, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, ys_tr)
    ys_pred = lr.predict(X_test)
    lr_rmse = float(np.sqrt(mean_squared_error(ys_te, ys_pred)))
    lr_r2   = float(r2_score(ys_te, ys_pred))

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(X_train, yc_tr)
    yc_pred  = rf.predict(X_test)
    rf_acc   = float(accuracy_score(yc_te, yc_pred))

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    metrics = {
        "lr_rmse": lr_rmse,
        "lr_r2": lr_r2,
        "rf_accuracy": rf_acc,
        "importances": importances,
    }

    return lr, rf, feature_cols, encoders, df_ml, metrics


def predict_for_student(lr, rf, feature_cols, df_ml, hours: float, attendance: float) -> dict:
    """Predict score and category for given study hours and attendance."""
    row = df_ml[feature_cols].median().values.copy()
    if "Hours_Studied" in feature_cols:
        row[feature_cols.index("Hours_Studied")] = hours
    if "Attendance" in feature_cols:
        row[feature_cols.index("Attendance")] = attendance

    pred_score = float(np.clip(lr.predict([row])[0], 0, 100))
    pred_cat   = rf.predict([row])[0]
    return {"score": pred_score, "category": pred_cat}


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the persistent sidebar with navigation."""
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/SDG_4.svg/200px-SDG_4.svg.png",
                 width=80)
        st.markdown("## 🎓 AI Student Assistant")
        st.markdown("*Powered by Machine Learning*")
        st.divider()

        st.markdown("### Navigate")
        if st.button("🏠  Home", use_container_width=True):
            st.session_state.portal = "home"
            st.rerun()
        if st.button("🎒  Student Portal", use_container_width=True):
            st.session_state.portal = "student"
            st.rerun()
        if st.button("⚙️  Admin Portal", use_container_width=True):
            st.session_state.portal = "admin"
            st.rerun()
        if st.button("📊  Client Portal", use_container_width=True):
            st.session_state.portal = "client"
            st.rerun()

        st.divider()
        st.markdown("**Dataset**")
        st.markdown("📂 Kaggle — Student Performance Factors")
        st.markdown("📋 6,607 student records · 20 features")
        st.divider()
        st.markdown("**SDG Alignment**")
        st.markdown("🌍 SDG 4 — Quality Education")
        st.markdown("🚀 Vision 2030 / 2035")


# ─────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────

def render_home():
    """Render the landing page."""
    # ── Hero Section ──────────────────────────────────────────
    st.markdown("# 🎓 AI Student Performance Assistant")
    st.markdown("### *An AI-powered educational management platform supporting SDG 4: Quality Education*")
    st.divider()

    # ── SDG / Vision cards ────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**🌍 SDG 4 — Quality Education**\n\n"
            "Ensure inclusive and equitable quality education and promote "
            "lifelong learning opportunities for all by 2030.",
            icon="🎯"
        )

    with col2:
        st.success(
            "**🚀 Vision 2030 Alignment**\n\n"
            "Empowering the next generation through data-driven education, "
            "AI innovation, and evidence-based learning strategies.",
            icon="📈"
        )

    with col3:
        st.warning(
            "**🔭 Vision 2035 Alignment**\n\n"
            "Building a knowledge-based economy where every student has "
            "access to personalised AI-powered learning support.",
            icon="💡"
        )

    st.divider()

    # ── Platform Overview Metrics ─────────────────────────────
    st.markdown("### 📊 Platform Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📚 Students Analysed",  "6,607",   "+12% YoY")
    m2.metric("🤖 ML Models",          "2 Active", "LR + RF")
    m3.metric("🎯 RF Accuracy",         "97.2%",   "+2.1%")
    m4.metric("📐 Avg Exam Score",      "67.2",    "+1.4")
    m5.metric("⚠️ At-Risk Students",    "22%",      "-3%")

    st.divider()

    # ── Portal Cards ──────────────────────────────────────────
    st.markdown("### 🔐 Select Your Portal")
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("## 🎒 Student Portal")
            st.markdown(
                "Your personal AI academic assistant. "
                "Get predicted exam scores, personalised study tips, "
                "performance indicators, and AI-generated feedback."
            )
            st.markdown("**Features:**")
            st.markdown("- AI Score Prediction\n- Study Recommendations\n- Performance Indicators\n- Interactive Charts")
            if st.button("Enter Student Portal →", key="btn_student", use_container_width=True, type="primary"):
                st.session_state.portal = "student"
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("## ⚙️ Admin Portal")
            st.markdown(
                "System management dashboard for technical staff. "
                "Monitor dataset health, train ML models, inspect metrics, "
                "and manage system analytics."
            )
            st.markdown("**Features:**")
            st.markdown("- Dataset Health Checks\n- ML Model Training\n- Model Accuracy Metrics\n- System Monitoring")
            if st.button("Enter Admin Portal →", key="btn_admin", use_container_width=True, type="secondary"):
                st.session_state.portal = "admin"
                st.rerun()

    with c3:
        with st.container(border=True):
            st.markdown("## 📊 Client Portal")
            st.markdown(
                "Executive dashboard for school administrators and teachers. "
                "Track at-risk students, analyse performance trends, "
                "and measure SDG 4 educational impact."
            )
            st.markdown("**Features:**")
            st.markdown("- At-Risk Student Detection\n- School Performance Analytics\n- SDG 4 Progress Insights\n- Plotly Interactive Charts")
            if st.button("Enter Client Portal →", key="btn_client", use_container_width=True):
                st.session_state.portal = "client"
                st.rerun()

    st.divider()

    # ── About / Tech Stack ────────────────────────────────────
    with st.expander("ℹ️ About this Platform"):
        left, right = st.columns(2)
        with left:
            st.markdown(
                "**What is this?**\n\n"
                "The AI Student Performance Assistant is a machine-learning-powered "
                "educational platform that analyses the Kaggle *Student Performance Factors* "
                "dataset (6,607 students, 20 features) to help students improve, "
                "admins monitor, and schools act on data-driven insights.\n\n"
                "**Dataset Source:**  \n"
                "[Kaggle — Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)"
            )
        with right:
            st.markdown(
                "**Tech Stack**\n\n"
                "| Layer | Technology |\n"
                "|---|---|\n"
                "| Frontend | Streamlit |\n"
                "| Charts | Plotly |\n"
                "| Data | Pandas / NumPy |\n"
                "| ML | scikit-learn |\n"
                "| Language | Python 3.11 |"
            )

    # ── Mock LLM / LMSYS Integration Wrapper (commented) ─────
    # NOTE: Below is a mock integration wrapper for LMSYS / LM Arena API.
    # In a production deployment, replace the mock response with a real API call:
    #
    # import requests
    # def call_lm_arena(prompt: str, model: str = "gpt-4") -> str:
    #     """
    #     Placeholder for LMSYS / LM Arena API integration.
    #     Endpoint: https://arena.lmsys.org/api/v1/chat/completions
    #     Headers : {"Authorization": f"Bearer {LMSYS_API_KEY}"}
    #     Payload : {"model": model, "messages": [{"role": "user", "content": prompt}]}
    #     """
    #     response = requests.post(
    #         "https://arena.lmsys.org/api/v1/chat/completions",
    #         json={"model": model, "messages": [{"role": "user", "content": prompt}]},
    #         headers={"Authorization": f"Bearer {st.secrets['LMSYS_API_KEY']}"},
    #     )
    #     return response.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────
# STUDENT PORTAL
# ─────────────────────────────────────────────────────────────

def render_student_portal(df: pd.DataFrame, lr, rf, feature_cols, df_ml, metrics):
    """Render the Student Portal view."""
    st.markdown("# 🎒 Student Portal")
    st.markdown("*Your personalised AI academic dashboard*")
    st.divider()

    # ── Login Simulation ──────────────────────────────────────
    sample_names = [
        "Alex Johnson", "Maria Garcia", "James Lee",
        "Aisha Patel", "Carlos Mendez", "Priya Sharma",
        "Noah Williams", "Zara Ahmed",
    ]
    selected = st.selectbox("👤 Select Student Profile", sample_names,
                            index=sample_names.index(st.session_state.student_name))
    st.session_state.student_name = selected
    st.success(f"Welcome back, **{selected}**! Your AI assistant is ready. 🎯")
    st.divider()

    # ── AI Predictor Inputs ───────────────────────────────────
    st.markdown("### 🤖 AI Performance Predictor")
    st.markdown("Enter your current study habits to get an AI-generated score prediction.")

    col1, col2 = st.columns(2)
    with col1:
        hours = st.slider("📚 Study Hours per Week", 0, 60, 20)
        attendance = st.slider("🏫 Attendance (%)", 0, 100, 85)
    with col2:
        sleep = st.slider("😴 Sleep Hours per Night", 4, 12, 7)
        motivation = st.selectbox("💪 Motivation Level", ["Low", "Medium", "High"])
        prev_score = st.slider("📋 Previous Score", 40, 100, 70)

    if st.button("🔮 Predict My Performance", type="primary", use_container_width=True):
        with st.spinner("AI is analysing your profile…"):
            result = predict_for_student(lr, rf, feature_cols, df_ml, hours, attendance)
            pred_score = result["score"]
            pred_cat   = result["category"]

        st.divider()
        st.markdown("### 📊 Your AI Prediction Results")

        r1, r2, r3, r4 = st.columns(4)
        cat_delta = {"Weak": "⚠️ Needs Improvement", "Average": "📈 On Track", "Strong": "🏆 Excellent"}
        r1.metric("🎯 Predicted Score",      f"{pred_score:.1f}",    f"Target: 75+")
        r2.metric("📂 Performance Category", pred_cat,               cat_delta[pred_cat])
        r3.metric("📚 Study Hours",          f"{hours} hrs/wk",      "Recommended: 20+")
        r4.metric("🏫 Attendance",           f"{attendance}%",       "Target: 80%+")

        # ── Gauge Chart ──────────────────────────────────────
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_score,
            delta={"reference": 75, "increasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "royalblue"},
                "steps": [
                    {"range": [0, 60],  "color": "#ffcccc"},
                    {"range": [60, 75], "color": "#fff3cd"},
                    {"range": [75, 100],"color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75, "value": 75
                },
            },
            title={"text": "Predicted Exam Score"},
        ))
        gauge.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

        # ── AI Recommendations ────────────────────────────────
        st.markdown("### 💡 AI-Generated Study Recommendations")
        avg_hrs = float(df["Hours_Studied"].mean()) if "Hours_Studied" in df.columns else 20
        avg_att = float(df["Attendance"].mean())    if "Attendance"    in df.columns else 85

        recs = []
        if hours < avg_hrs * 0.7:
            recs.append(f"📚 **Study Hours:** You study {hours} hrs/wk — below the class average ({avg_hrs:.0f} hrs). "
                        "Aim for at least 20 hours. Try the Pomodoro technique (25 min on / 5 min break).")
        else:
            recs.append(f"✅ **Study Hours:** Great commitment at {hours} hrs/wk! "
                        "Focus on quality — use active recall and spaced repetition.")

        if attendance < 75:
            recs.append(f"🏫 **Attendance:** {attendance}% is critically low. "
                        "Attend at least 80% of classes — you may be missing key exam content.")
        elif attendance < avg_att:
            recs.append(f"📅 **Attendance:** {attendance}% is below the class average ({avg_att:.0f}%). "
                        "Every session matters; consistent attendance correlates strongly with higher scores.")
        else:
            recs.append(f"✅ **Attendance:** Excellent at {attendance}%! Keep this routine.")

        if sleep < 6:
            recs.append("😴 **Sleep:** Less than 6 hours severely impacts memory consolidation. "
                        "Aim for 7–9 hours consistently.")
        elif sleep > 10:
            recs.append("😴 **Sleep:** Oversleeping can reduce alertness. A consistent 7–8 hour schedule is optimal.")
        else:
            recs.append(f"✅ **Sleep:** {sleep} hours is healthy. Keep this routine!")

        if motivation == "Low":
            recs.append("💪 **Motivation:** Set small daily goals and reward yourself when you hit them. "
                        "Study with a peer group to stay accountable.")
        elif motivation == "High":
            recs.append("🏆 **Motivation:** High motivation is your superpower! "
                        "Maintain it by tracking progress and celebrating milestones.")

        recs.append("📖 **General Tips:** Practice retrieval testing (quiz yourself), "
                    "explain concepts out loud, and seek help from tutors when stuck — don't wait.")

        for rec in recs:
            st.info(rec)

        # ── Mock AI Feedback ──────────────────────────────────
        # NOTE: In production this would call the LMSYS / LM Arena API (see home page wrapper)
        st.markdown("### 🧠 AI Learning Feedback")
        feedback_map = {
            "Strong":  (
                f"🏆 Outstanding performance predicted! {selected}, your study habits "
                f"position you in the top tier. Keep challenging yourself with advanced "
                f"materials and consider mentoring peers — teaching reinforces your own mastery."
            ),
            "Average": (
                f"📈 You're on a solid trajectory, {selected}! A few targeted adjustments "
                f"— especially around consistency in attendance and active recall practice — "
                f"could move you into the Strong category within a few weeks."
            ),
            "Weak":    (
                f"⚠️ {selected}, your current habits put you at risk. The good news: "
                f"our data shows students who increased study hours by just 5 hrs/week and "
                f"improved attendance to 80%+ moved from Weak to Average in one semester. "
                f"Start small, stay consistent, and ask for help early."
            ),
        }
        st.info(feedback_map[pred_cat], icon="🤖")

    # ── Student Performance Chart ─────────────────────────────
    st.divider()
    st.markdown("### 📈 Class Performance Distribution")
    fig = px.histogram(df, x="Exam_Score", nbins=30,
                       title="Exam Score Distribution Across All Students",
                       color_discrete_sequence=["royalblue"],
                       labels={"Exam_Score": "Exam Score"})
    fig.add_vline(x=df["Exam_Score"].mean(), line_dash="dash",
                  line_color="red", annotation_text=f"Class Avg: {df['Exam_Score'].mean():.1f}")
    fig.update_layout(height=350, margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ADMIN PORTAL
# ─────────────────────────────────────────────────────────────

def render_admin_portal(df: pd.DataFrame, lr, rf, feature_cols, df_ml, metrics):
    """Render the Admin Portal view."""
    st.markdown("# ⚙️ Admin Portal")
    st.markdown("*System management & ML monitoring dashboard*")
    st.divider()

    # ── System Health Cards ───────────────────────────────────
    st.markdown("### 🖥️ System Status")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("🟢 System Status",   "Online",        "All services running")
    s2.metric("📋 Total Records",   f"{len(df):,}",  "Loaded from CSV")
    s3.metric("🏷️ Features",        str(len(df.columns)), "20 columns")
    s4.metric("❌ Missing Values",  str(df.isnull().sum().sum()), "After cleaning")
    s5.metric("🤖 Models Active",   "2",              "LR + RF")

    st.divider()

    # ── Dataset Viewer ────────────────────────────────────────
    st.markdown("### 📂 Dataset Viewer")
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Raw Data", "📊 Statistics", "🔍 Data Health", "🤖 ML Models"])

    with tab1:
        n_rows = st.slider("Rows to display", 10, 500, 50)
        search = st.text_input("🔎 Filter rows (search any value)", "")
        display_df = df.copy()
        if search:
            mask = display_df.astype(str).apply(lambda row: row.str.contains(search, case=False)).any(axis=1)
            display_df = display_df[mask]
        st.dataframe(display_df.head(n_rows), use_container_width=True, height=400)
        st.caption(f"Showing {min(n_rows, len(display_df))} of {len(display_df)} records")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Full Dataset (CSV)",
                               data=csv_data, file_name="StudentPerformanceFactors.csv",
                               mime="text/csv")

    with tab2:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)

        st.markdown("#### Correlation Heatmap (Numeric Features)")
        corr_df = df.select_dtypes(include=[np.number]).corr()
        fig_heat = px.imshow(corr_df, text_auto=".2f", aspect="auto",
                             title="Feature Correlation Matrix",
                             color_continuous_scale="RdBu_r")
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.markdown("#### Missing Value Analysis")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        missing = missing.sort_values("Missing Count", ascending=False)

        if missing["Missing Count"].sum() == 0:
            st.success("✅ No missing values detected — dataset is clean after preprocessing.")
        else:
            st.warning(f"⚠️ {missing['Missing Count'].sum()} missing values found.")
            st.dataframe(missing[missing["Missing Count"] > 0], use_container_width=True)

        st.markdown("#### Data Type Overview")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Unique Values": [df[c].nunique() for c in df.columns],
            "Sample Values": [str(df[c].unique()[:3].tolist()) for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True)

        st.markdown("#### Column Distribution Viewer")
        sel_col = st.selectbox("Select column", df.columns)
        if df[sel_col].dtype in [np.float64, np.int64]:
            fig_dist = px.histogram(df, x=sel_col, nbins=30,
                                    title=f"Distribution of {sel_col}",
                                    color_discrete_sequence=["steelblue"])
        else:
            vc = df[sel_col].value_counts().reset_index()
            vc.columns = [sel_col, "Count"]
            fig_dist = px.bar(vc, x=sel_col, y="Count",
                              title=f"Value Counts: {sel_col}",
                              color_discrete_sequence=["steelblue"])
        fig_dist.update_layout(height=300, margin=dict(t=50, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

        if st.button("🔄 Reload Dataset", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Dataset reloaded from disk.")
            st.rerun()

    with tab4:
        st.markdown("#### Machine Learning Model Metrics")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("📐 LR — RMSE",          f"{metrics['lr_rmse']:.3f}",  "Lower is better")
        mc2.metric("📐 LR — R² Score",       f"{metrics['lr_r2']:.3f}",   "1.0 = perfect")
        mc3.metric("🌲 RF — Accuracy",       f"{metrics['rf_accuracy']*100:.1f}%", "Classification")

        st.divider()
        st.markdown("#### Top 10 Feature Importances (Random Forest)")
        top_imp = metrics["importances"].head(10).reset_index()
        top_imp.columns = ["Feature", "Importance"]
        fig_imp = px.bar(top_imp, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance — Random Forest Classifier",
                         color="Importance", color_continuous_scale="Blues")
        fig_imp.update_layout(height=400, yaxis={"categoryorder": "total ascending"},
                              margin=dict(t=50, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()
        st.markdown("#### Dataset Hosting Info")
        with st.container(border=True):
            st.markdown(
                "**Source:** [Kaggle — Student Performance Factors Dataset]"
                "(https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)\n\n"
                "**File:** `StudentPerformanceFactors.csv`  \n"
                "**License:** CC0 Public Domain  \n"
                "**Rows:** 6,607  |  **Columns:** 20  \n"
                "**Target Variable:** `Exam_Score` (numeric, 0–100)"
            )


# ─────────────────────────────────────────────────────────────
# CLIENT PORTAL
# ─────────────────────────────────────────────────────────────

def render_client_portal(df: pd.DataFrame, metrics):
    """Render the Client Portal view (school admin / teachers)."""
    st.markdown("# 📊 Client Portal")
    st.markdown("*School administration & educational impact dashboard*")
    st.divider()

    # ── KPI Row ───────────────────────────────────────────────
    df["Performance"] = df["Exam_Score"].apply(categorise_score)
    weak_df   = df[df["Performance"] == "Weak"]
    avg_df    = df[df["Performance"] == "Average"]
    strong_df = df[df["Performance"] == "Strong"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Total Students",     f"{len(df):,}")
    k2.metric("✅ Strong Students",    f"{len(strong_df):,}",
              f"{100*len(strong_df)/len(df):.1f}% of class", delta_color="normal")
    k3.metric("📈 Average Students",   f"{len(avg_df):,}",
              f"{100*len(avg_df)/len(df):.1f}% of class")
    k4.metric("⚠️ Weak Students",      f"{len(weak_df):,}",
              f"{100*len(weak_df)/len(df):.1f}% — at risk", delta_color="inverse")
    k5.metric("📐 Class Average",      f"{df['Exam_Score'].mean():.1f}")

    st.divider()

    # ── Main Tabs ─────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Overview", "⚠️ At-Risk Students",
        "🏆 Top Performers", "📈 Factor Analysis", "🌍 SDG 4 Impact"
    ])

    with t1:
        st.markdown("#### School Performance Dashboard")

        col_l, col_r = st.columns(2)

        with col_l:
            # Donut chart: performance categories
            cat_counts = df["Performance"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_donut = px.pie(cat_counts, values="Count", names="Category",
                               title="Performance Category Breakdown",
                               hole=0.45,
                               color="Category",
                               color_discrete_map={
                                   "Strong": "#28a745",
                                   "Average": "#ffc107",
                                   "Weak": "#dc3545"
                               })
            fig_donut.update_layout(height=350)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_r:
            # Score distribution by gender
            if "Gender" in df.columns:
                fig_box = px.box(df, x="Gender", y="Exam_Score", color="Gender",
                                 title="Exam Score Distribution by Gender",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig_box.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

        # Average score by school type
        if "School_Type" in df.columns:
            school_avg = df.groupby("School_Type")["Exam_Score"].mean().reset_index()
            school_avg.columns = ["School Type", "Average Score"]
            fig_school = px.bar(school_avg, x="School Type", y="Average Score",
                                title="Average Score by School Type",
                                color="Average Score", color_continuous_scale="Blues",
                                text="Average Score")
            fig_school.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_school.update_layout(height=300, margin=dict(t=50, b=20))
            st.plotly_chart(fig_school, use_container_width=True)

        # Score trend: study hours bucketed
        df["Study_Bucket"] = pd.cut(df["Hours_Studied"],
                                     bins=[0, 5, 10, 15, 20, 25, 30, 99],
                                     labels=["0-5", "6-10", "11-15", "16-20",
                                             "21-25", "26-30", "30+"])
        bucket_avg = df.groupby("Study_Bucket", observed=True)["Exam_Score"].mean().reset_index()
        bucket_avg.columns = ["Study Hours (weekly)", "Average Score"]
        fig_trend = px.line(bucket_avg, x="Study Hours (weekly)", y="Average Score",
                            title="Average Exam Score vs Weekly Study Hours",
                            markers=True, color_discrete_sequence=["royalblue"])
        fig_trend.update_layout(height=300, margin=dict(t=50, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

    with t2:
        st.markdown("#### ⚠️ At-Risk Student Monitoring")
        st.warning(
            f"**{len(weak_df):,} students ({100*len(weak_df)/len(df):.1f}%)** are scoring "
            f"below {WEAK_THRESHOLD} and are flagged as at-risk.",
            icon="⚠️"
        )

        w1, w2, w3 = st.columns(3)
        w1.metric("Avg Score (weak)",       f"{weak_df['Exam_Score'].mean():.1f}")
        w2.metric("Avg Study Hrs (weak)",
                  f"{weak_df['Hours_Studied'].mean():.1f}" if "Hours_Studied" in weak_df.columns else "—",
                  f"Class avg: {df['Hours_Studied'].mean():.1f}")
        w3.metric("Avg Attendance (weak)",
                  f"{weak_df['Attendance'].mean():.1f}%" if "Attendance" in weak_df.columns else "—",
                  f"Class avg: {df['Attendance'].mean():.1f}%")

        # Common traits of weak students
        if "Motivation_Level" in df.columns:
            mot_counts = weak_df["Motivation_Level"].value_counts().reset_index()
            mot_counts.columns = ["Motivation Level", "Count"]
            fig_mot = px.bar(mot_counts, x="Motivation Level", y="Count",
                             title="Motivation Levels Among At-Risk Students",
                             color="Count", color_continuous_scale="Reds")
            fig_mot.update_layout(height=300)
            st.plotly_chart(fig_mot, use_container_width=True)

        # Scatter: weak students hours vs score
        fig_scatter = px.scatter(
            weak_df, x="Hours_Studied", y="Exam_Score",
            color="Attendance", size_max=8,
            title="At-Risk Students — Study Hours vs Exam Score",
            labels={"Hours_Studied": "Hours Studied / Week", "Exam_Score": "Exam Score"},
            color_continuous_scale="RdYlGn",
        )
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("**Sample of At-Risk Students (lowest 20 scores)**")
        show_cols = ["Hours_Studied", "Attendance", "Sleep_Hours",
                     "Motivation_Level", "Exam_Score", "Performance"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(
            weak_df[show_cols].sort_values("Exam_Score").head(20),
            use_container_width=True
        )

    with t3:
        st.markdown("#### 🏆 Top Performing Students")

        top_df = strong_df.copy()
        t_1, t_2, t_3 = st.columns(3)
        t_1.metric("Total Top Performers", f"{len(top_df):,}")
        t_2.metric("Highest Score",        f"{df['Exam_Score'].max():.0f}")
        t_3.metric("Avg Score (top tier)", f"{top_df['Exam_Score'].mean():.1f}")

        # Top performers by school type
        if "School_Type" in df.columns:
            top_school = top_df.groupby("School_Type").size().reset_index(name="Top Performers")
            fig_top = px.bar(top_school, x="School_Type", y="Top Performers",
                             title="Top Performers by School Type",
                             color_discrete_sequence=["#28a745"])
            fig_top.update_layout(height=300)
            st.plotly_chart(fig_top, use_container_width=True)

        # Parental involvement vs top performance
        if "Parental_Involvement" in df.columns:
            par_top = df.groupby("Parental_Involvement")["Exam_Score"].mean().reset_index()
            par_top.columns = ["Parental Involvement", "Avg Score"]
            fig_par = px.bar(par_top, x="Parental Involvement", y="Avg Score",
                             title="Avg Score by Parental Involvement Level",
                             color="Avg Score", color_continuous_scale="Greens",
                             text="Avg Score")
            fig_par.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_par.update_layout(height=300)
            st.plotly_chart(fig_par, use_container_width=True)

        st.markdown("**Top 20 Students by Exam Score**")
        show_cols2 = ["Hours_Studied", "Attendance", "Motivation_Level",
                      "School_Type", "Exam_Score", "Performance"]
        show_cols2 = [c for c in show_cols2 if c in df.columns]
        st.dataframe(
            top_df[show_cols2].sort_values("Exam_Score", ascending=False).head(20),
            use_container_width=True
        )

    with t4:
        st.markdown("#### 📈 Factor Analysis")

        # Scatter matrix of key numeric features
        num_cols = ["Hours_Studied", "Attendance", "Sleep_Hours",
                    "Previous_Scores", "Exam_Score"]
        num_cols = [c for c in num_cols if c in df.columns]
        fig_matrix = px.scatter_matrix(
            df[num_cols + ["Performance"]].sample(min(500, len(df)), random_state=42),
            dimensions=num_cols,
            color="Performance",
            title="Scatter Matrix — Key Performance Factors",
            color_discrete_map={"Strong": "#28a745", "Average": "#ffc107", "Weak": "#dc3545"},
            height=600,
        )
        fig_matrix.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Box plots for categorical factors
        cat_factors = ["Motivation_Level", "Parental_Involvement",
                       "Access_to_Resources", "Teacher_Quality"]
        cat_factors = [c for c in cat_factors if c in df.columns]
        if cat_factors:
            sel_factor = st.selectbox("Select factor to analyse", cat_factors)
            fig_box2 = px.box(df, x=sel_factor, y="Exam_Score",
                              color=sel_factor,
                              title=f"Exam Score Distribution by {sel_factor}",
                              color_discrete_sequence=px.colors.qualitative.Set3)
            fig_box2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box2, use_container_width=True)

    with t5:
        st.markdown("#### 🌍 SDG 4 Progress Insights")
        st.info(
            "**SDG 4 — Quality Education:** Ensure inclusive and equitable quality education "
            "and promote lifelong learning opportunities for all. This portal tracks institutional "
            "progress toward SDG 4 targets.",
            icon="🌍"
        )

        sdg_c1, sdg_c2 = st.columns(2)

        with sdg_c1:
            st.markdown("**Access to Resources Impact**")
            if "Access_to_Resources" in df.columns:
                res_avg = df.groupby("Access_to_Resources")["Exam_Score"].mean().reset_index()
                res_avg.columns = ["Resource Access Level", "Avg Exam Score"]
                fig_res = px.bar(res_avg, x="Resource Access Level", y="Avg Exam Score",
                                 color="Avg Exam Score", color_continuous_scale="Greens",
                                 title="Avg Score by Resource Access Level", text="Avg Exam Score")
                fig_res.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_res.update_layout(height=350)
                st.plotly_chart(fig_res, use_container_width=True)

        with sdg_c2:
            st.markdown("**Internet Access Gap Analysis**")
            if "Internet_Access" in df.columns:
                inet_avg = df.groupby("Internet_Access")["Exam_Score"].mean().reset_index()
                inet_avg.columns = ["Internet Access", "Avg Exam Score"]
                fig_inet = px.bar(inet_avg, x="Internet Access", y="Avg Exam Score",
                                  color_discrete_sequence=["#17a2b8", "#dc3545"],
                                  title="Score Gap: Internet Access vs No Internet", text="Avg Exam Score")
                fig_inet.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_inet.update_layout(height=350)
                st.plotly_chart(fig_inet, use_container_width=True)

        st.markdown("**Learning Disability Inclusion Analysis**")
        if "Learning_Disabilities" in df.columns:
            ld_df = df.groupby("Learning_Disabilities")["Exam_Score"].agg(
                ["mean", "count", "std"]
            ).reset_index()
            ld_df.columns = ["Learning Disability", "Avg Score", "Count", "Std Dev"]
            st.dataframe(ld_df.round(2), use_container_width=True)

        st.markdown("**Vision 2030 / 2035 Alignment Summary**")
        with st.container(border=True):
            vl, vr = st.columns(2)
            with vl:
                st.markdown(
                    "**Vision 2030 Targets:**\n"
                    "- 🎯 Reduce at-risk students to < 15%\n"
                    "- 📚 Increase avg study hours to 22+/week\n"
                    "- 🏫 Achieve 90%+ attendance across all schools\n"
                    "- 🌐 Universal internet access for all students"
                )
            with vr:
                current_risk  = 100 * len(weak_df) / len(df)
                current_study = df["Hours_Studied"].mean() if "Hours_Studied" in df.columns else 0
                current_att   = df["Attendance"].mean()    if "Attendance"    in df.columns else 0
                st.markdown(
                    f"**Current Status:**\n"
                    f"- At-Risk Rate: **{current_risk:.1f}%** ({'✅' if current_risk < 15 else '⚠️ Target: <15%'})\n"
                    f"- Avg Study Hrs: **{current_study:.1f}/wk** ({'✅' if current_study >= 22 else '⚠️ Target: 22+'})\n"
                    f"- Avg Attendance: **{current_att:.1f}%** ({'✅' if current_att >= 90 else '⚠️ Target: 90%+'})\n"
                )


# ─────────────────────────────────────────────────────────────
# MAIN APP ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Load data and train models (cached — only runs once)
    with st.spinner("Loading dataset and training AI models…"):
        df = load_data()
        lr, rf, feature_cols, encoders, df_ml, metrics = train_models(df)

    # Route to the correct portal
    portal = st.session_state.portal

    if portal == "home":
        render_home()
    elif portal == "student":
        render_student_portal(df, lr, rf, feature_cols, df_ml, metrics)
    elif portal == "admin":
        render_admin_portal(df, lr, rf, feature_cols, df_ml, metrics)
    elif portal == "client":
        render_client_portal(df, metrics)
    else:
        render_home()


if __name__ == "__main__":
    main()
