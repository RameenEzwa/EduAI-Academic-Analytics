# ============================================================
# AI Student Performance Assistant — Streamlit Web Application
# Supports SDG 4: Quality Education | Vision 2030 / 2035
# ============================================================
# Page flow:
#   Home  →  Choose Portal  →  Focused Dashboard
#
# Portals:
#   Student Portal  — prediction, recommendations, AI feedback
#   Admin Portal    — dataset management, ML monitoring
#   Teacher Portal  — school analytics, at-risk reports, SDG
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Student Performance Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DATASET_FILE   = "StudentPerformanceFactors.csv"
WEAK_THRESHOLD = 60
AVG_THRESHOLD  = 75

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page":         "home",   # home | student | admin | teacher
        "username":     "",
        "role":         "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def go_to(page: str, username: str = "", role: str = ""):
    st.session_state.page     = page
    if username:
        st.session_state.username = username
    if role:
        st.session_state.role     = role
    st.rerun()


def categorise(score: float) -> str:
    if score < WEAK_THRESHOLD:
        return "Weak"
    elif score < AVG_THRESHOLD:
        return "Average"
    return "Strong"


# ─────────────────────────────────────────────────────────────
# DATA & MODELS  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_FILE)
    df.dropna(subset=["Exam_Score"], inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    df["Performance"] = df["Exam_Score"].apply(categorise)
    return df


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    df_ml = df.drop(columns=["Performance"]).copy()
    encoders = {}
    for c in df_ml.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_ml[c] = le.fit_transform(df_ml[c].astype(str))
        encoders[c] = le

    feat_cols = [c for c in df_ml.columns if c != "Exam_Score"]
    X  = df_ml[feat_cols].values
    ys = df_ml["Exam_Score"].values
    yc = np.array([categorise(s) for s in ys])

    X_tr, X_te, ys_tr, ys_te, yc_tr, yc_te = train_test_split(
        X, ys, yc, test_size=0.2, random_state=42)

    lr = LinearRegression().fit(X_tr, ys_tr)
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(X_tr, yc_tr)

    metrics = {
        "rmse":     float(np.sqrt(mean_squared_error(ys_te, lr.predict(X_te)))),
        "r2":       float(r2_score(ys_te, lr.predict(X_te))),
        "accuracy": float(accuracy_score(yc_te, rf.predict(X_te))),
        "importances": pd.Series(rf.feature_importances_, index=feat_cols)
                          .sort_values(ascending=False),
    }
    return lr, rf, feat_cols, df_ml, metrics


def predict(lr, rf, feat_cols, df_ml, hours, attendance):
    row = df_ml[feat_cols].median().values.copy()
    if "Hours_Studied" in feat_cols:
        row[feat_cols.index("Hours_Studied")] = hours
    if "Attendance" in feat_cols:
        row[feat_cols.index("Attendance")]    = attendance
    score = float(np.clip(lr.predict([row])[0], 0, 100))
    cat   = rf.predict([row])[0]
    return score, cat


# ─────────────────────────────────────────────────────────────
# SIDEBAR  (minimal — shown only inside portals)
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    page = st.session_state.page
    if page == "home":
        return

    with st.sidebar:
        st.markdown("### 🎓 AI Student Assistant")
        st.divider()
        role_icons = {"student": "🎒", "admin": "⚙️", "teacher": "📊"}
        icon = role_icons.get(page, "👤")
        st.markdown(f"**{icon} Logged in as:**")
        st.markdown(f"*{st.session_state.username or 'Guest'}*")
        st.markdown(f"Role: `{st.session_state.role or page.title()}`")
        st.divider()

        if st.button("🏠  Back to Home", use_container_width=True):
            go_to("home")

        st.divider()
        st.caption("Dataset: Kaggle — Student Performance Factors")
        st.caption("SDG 4 · Vision 2030 / 2035")


# ─────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────
def render_home():
    # ── Hero ──────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center; padding-top:2rem;'>🎓 AI Student Performance Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:1.2rem; color:gray;'>"
        "An AI-powered educational management platform supporting "
        "<strong>SDG 4: Quality Education</strong> · Vision 2030 / 2035"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── SDG + Vision strip ────────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        st.info(
            "**🌍 SDG 4 — Quality Education**  \n"
            "Ensure inclusive, equitable quality education and promote "
            "lifelong learning for all.",
        )
    with sc2:
        st.success(
            "**🚀 Vision 2030 / 2035**  \n"
            "Empowering a knowledge-based economy through AI-driven "
            "personalised learning and data-informed teaching.",
        )

    st.divider()

    # ── Portal Cards ──────────────────────────────────────────
    st.markdown(
        "<h3 style='text-align:center;'>Select Your Portal</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("## 🎒")
            st.markdown("### Student Portal")
            st.markdown(
                "Get your AI-predicted exam score, personalised study "
                "recommendations, and academic improvement tips."
            )
            st.markdown("&nbsp;")
            name = st.text_input("Your name", placeholder="e.g. Alex Johnson",
                                 key="home_student_name")
            if st.button("Enter Student Portal", key="go_student",
                         use_container_width=True, type="primary"):
                go_to("student", username=name or "Student", role="Student")

    with c2:
        with st.container(border=True):
            st.markdown("## ⚙️")
            st.markdown("### Admin Portal")
            st.markdown(
                "Manage the dataset, monitor ML model health, view system "
                "analytics, and inspect data quality metrics."
            )
            st.markdown("&nbsp;")
            admin_id = st.text_input("Admin ID", placeholder="e.g. admin@school.edu",
                                     key="home_admin_id")
            if st.button("Enter Admin Portal", key="go_admin",
                         use_container_width=True, type="secondary"):
                go_to("admin", username=admin_id or "Admin", role="Administrator")

    with c3:
        with st.container(border=True):
            st.markdown("## 📊")
            st.markdown("### Teacher Portal")
            st.markdown(
                "Monitor at-risk students, analyse school performance, "
                "track SDG 4 progress, and generate class reports."
            )
            st.markdown("&nbsp;")
            teacher_id = st.text_input("Teacher name", placeholder="e.g. Ms. Rivera",
                                       key="home_teacher_id")
            if st.button("Enter Teacher Portal", key="go_teacher",
                         use_container_width=True):
                go_to("teacher", username=teacher_id or "Teacher", role="Teacher")

    st.divider()
    st.caption(
        "Dataset: [Kaggle — Student Performance Factors]"
        "(https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) · "
        "6,607 records · 20 features · CC0 License"
    )


# ─────────────────────────────────────────────────────────────
# STUDENT PORTAL
# ─────────────────────────────────────────────────────────────
def render_student(df, lr, rf, feat_cols, df_ml, metrics):
    name = st.session_state.username or "Student"
    st.markdown(f"# 🎒 Student Portal")
    st.markdown(f"Welcome, **{name}**! Enter your study habits below to receive your AI performance report.")
    st.divider()

    # ── Inputs ────────────────────────────────────────────────
    st.markdown("### 📋 Your Study Profile")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        hours      = st.slider("📚 Study hours per week",  0,  60, 20)
        attendance = st.slider("🏫 Attendance (%)",         0, 100, 85)
        sleep      = st.slider("😴 Sleep hours per night",  4,  12,  7)

    with col2:
        motivation  = st.selectbox("💪 Motivation level", ["Low", "Medium", "High"], index=1)
        prev_score  = st.slider("📋 Previous exam score", 40, 100, 70)
        has_internet = st.radio("🌐 Internet access", ["Yes", "No"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔮  Generate AI Performance Report", type="primary", use_container_width=True)

    if not run:
        return

    # ── Prediction ────────────────────────────────────────────
    with st.spinner("AI is analysing your profile…"):
        score, category = predict(lr, rf, feat_cols, df_ml, hours, attendance)

    st.divider()
    st.markdown("## 📊 Your AI Performance Report")

    # KPI row
    cat_label = {"Weak": "⚠️ Needs Improvement", "Average": "📈 On Track", "Strong": "🏆 Excellent"}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🎯 Predicted Score",      f"{score:.1f} / 100")
    k2.metric("📂 Performance Category", category,         cat_label[category])
    k3.metric("📚 Study Hours",          f"{hours} hrs/wk",  "Recommended: 20+")
    k4.metric("🏫 Attendance",           f"{attendance}%",    "Target: 80%+")

    # Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 75, "increasing": {"color": "green"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "royalblue"},
            "steps": [
                {"range": [0, WEAK_THRESHOLD],               "color": "#ffcccc"},
                {"range": [WEAK_THRESHOLD, AVG_THRESHOLD],   "color": "#fff3cd"},
                {"range": [AVG_THRESHOLD,  100],             "color": "#d4edda"},
            ],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.75, "value": 75},
        },
        title={"text": "Predicted Exam Score"},
    ))
    gauge.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)

    st.divider()

    # ── Recommendations ───────────────────────────────────────
    st.markdown("### 💡 Personalised Study Recommendations")
    avg_hrs = float(df["Hours_Studied"].mean())
    avg_att = float(df["Attendance"].mean())

    recs = []
    if hours < avg_hrs * 0.7:
        recs.append(("📚 Study Hours",
                      f"You study {hours} hrs/wk — below the class average ({avg_hrs:.0f} hrs). "
                      "Aim for 20+ hours. Use the Pomodoro technique (25 min on / 5 min break)."))
    else:
        recs.append(("✅ Study Hours",
                      f"Good commitment at {hours} hrs/wk. "
                      "Focus on quality — use active recall and spaced repetition."))

    if attendance < 75:
        recs.append(("🏫 Attendance",
                      f"{attendance}% is critically low. Target 80%+ to avoid missing key content."))
    elif attendance < avg_att:
        recs.append(("📅 Attendance",
                      f"{attendance}% is below the class average ({avg_att:.0f}%). "
                      "Every session matters."))
    else:
        recs.append(("✅ Attendance", f"Excellent at {attendance}%! Keep it up."))

    if sleep < 6:
        recs.append(("😴 Sleep",
                      "Less than 6 hours severely impacts memory consolidation. Aim for 7–9 hours."))
    elif sleep > 10:
        recs.append(("😴 Sleep",
                      "Oversleeping can reduce alertness. A consistent 7–8 hour schedule is optimal."))
    else:
        recs.append(("✅ Sleep", f"{sleep} hours is healthy. Maintain this routine."))

    if motivation == "Low":
        recs.append(("💪 Motivation",
                      "Set small daily goals and reward yourself when you reach them. "
                      "Study with a peer group to stay accountable."))
    elif motivation == "High":
        recs.append(("🏆 Motivation",
                      "High motivation is your superpower! Track your progress and celebrate milestones."))

    for title, body in recs:
        with st.expander(title, expanded=True):
            st.write(body)

    st.divider()

    # ── AI Feedback ───────────────────────────────────────────
    st.markdown("### 🤖 AI Learning Feedback")
    feedback = {
        "Strong":  (
            f"🏆 **Outstanding!** {name}, your predicted score of {score:.1f} places you in the top tier. "
            "Keep challenging yourself with advanced materials and consider mentoring peers — "
            "teaching reinforces your own mastery."
        ),
        "Average": (
            f"📈 **On track, {name}!** A predicted score of {score:.1f} is solid. "
            "A few targeted adjustments — particularly consistent attendance and active recall — "
            "could push you into the Strong category within weeks."
        ),
        "Weak": (
            f"⚠️ **{name}, action is needed.** Your predicted score of {score:.1f} is below the pass threshold. "
            "Students who added just 5 study hours per week and brought attendance to 80%+ improved "
            "by a full category in one semester. Start small, stay consistent, ask for help early."
        ),
    }
    st.info(feedback[category], icon="🤖")

    # ── Class context chart ────────────────────────────────────
    st.divider()
    st.markdown("### 📈 Your Score in Class Context")
    fig = px.histogram(df, x="Exam_Score", nbins=30,
                       color_discrete_sequence=["#6ea8fe"],
                       labels={"Exam_Score": "Exam Score"},
                       title="Class Score Distribution — Your Prediction Highlighted")
    fig.add_vline(x=score, line_color="red", line_dash="solid", line_width=3,
                  annotation_text=f"You: {score:.1f}", annotation_position="top right")
    fig.add_vline(x=df["Exam_Score"].mean(), line_color="orange", line_dash="dash",
                  annotation_text=f"Class avg: {df['Exam_Score'].mean():.1f}")
    fig.update_layout(height=320, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ADMIN PORTAL
# ─────────────────────────────────────────────────────────────
def render_admin(df, lr, rf, feat_cols, df_ml, metrics):
    st.markdown("# ⚙️ Admin Portal")
    st.markdown("*System management · Dataset health · ML model monitoring*")
    st.divider()

    # ── System status ─────────────────────────────────────────
    st.markdown("### 🖥️ System Status")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Status",          "🟢 Online")
    s2.metric("Records",         f"{len(df):,}")
    s3.metric("Features",        str(len(df.columns) - 1))
    s4.metric("Missing Values",  str(df.drop(columns=["Performance"]).isnull().sum().sum()))
    s5.metric("Active Models",   "2  (LR + RF)")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📊 Data Analysis", "🤖 ML Models"])

    # ── Tab 1: Dataset viewer ─────────────────────────────────
    with tab1:
        st.markdown("#### Dataset Viewer")
        search = st.text_input("🔎 Filter by any value", "")
        display = df.copy()
        if search:
            mask    = display.astype(str).apply(lambda r: r.str.contains(search, case=False)).any(axis=1)
            display = display[mask]
        n = st.slider("Rows to display", 10, 500, 50)
        st.dataframe(display.head(n), use_container_width=True, height=400)
        st.caption(f"Showing {min(n, len(display))} of {len(display)} records")
        st.download_button(
            "⬇️ Download CSV",
            data=df.drop(columns=["Performance"]).to_csv(index=False).encode(),
            file_name="StudentPerformanceFactors.csv",
            mime="text/csv",
        )

        if st.button("🔄 Reload Dataset from Disk"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared — reload the page to re-fetch data.")

    # ── Tab 2: Data analysis ──────────────────────────────────
    with tab2:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.select_dtypes(include=[np.number]).describe().round(2),
                     use_container_width=True)

        st.markdown("#### Missing Value Check")
        miss = df.isnull().sum()
        if miss.sum() == 0:
            st.success("✅ No missing values after preprocessing.")
        else:
            st.warning(f"⚠️ {miss.sum()} missing values remain.")
            st.dataframe(miss[miss > 0].rename("Missing Count"), use_container_width=True)

        st.markdown("#### Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r",
                             title="Feature Correlation Matrix")
        fig_heat.update_layout(height=480)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Column Distribution Explorer")
        col_sel = st.selectbox("Select column", df.columns)
        if df[col_sel].dtype in [np.float64, np.int64]:
            fig_col = px.histogram(df, x=col_sel, nbins=30,
                                   color_discrete_sequence=["steelblue"],
                                   title=f"Distribution: {col_sel}")
        else:
            vc = df[col_sel].value_counts().reset_index()
            vc.columns = [col_sel, "Count"]
            fig_col = px.bar(vc, x=col_sel, y="Count",
                             color_discrete_sequence=["steelblue"],
                             title=f"Value Counts: {col_sel}")
        fig_col.update_layout(height=300)
        st.plotly_chart(fig_col, use_container_width=True)

    # ── Tab 3: ML Models ──────────────────────────────────────
    with tab3:
        st.markdown("#### Model Performance Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("LR — RMSE",    f"{metrics['rmse']:.3f}",     "Lower = better")
        m2.metric("LR — R² Score",f"{metrics['r2']:.3f}",       "1.0 = perfect")
        m3.metric("RF — Accuracy",f"{metrics['accuracy']*100:.1f}%", "Category classification")

        st.markdown("#### Top 10 Feature Importances")
        top10 = metrics["importances"].head(10).reset_index()
        top10.columns = ["Feature", "Importance"]
        fig_imp = px.bar(top10, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues",
                         title="Random Forest Feature Importances")
        fig_imp.update_layout(height=380,
                              yaxis={"categoryorder": "total ascending"},
                              margin=dict(t=50, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Dataset Source")
        with st.container(border=True):
            st.markdown(
                "**Source:** [Kaggle — Student Performance Factors]"
                "(https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)  \n"
                "**File:** `StudentPerformanceFactors.csv`  \n"
                "**License:** CC0 Public Domain  \n"
                "**Records:** 6,607  |  **Features:** 20  \n"
                "**Target:** `Exam_Score` (0–100)"
            )

        # Mock LLM integration wrapper (commented — ready for production)
        # ──────────────────────────────────────────────────────
        # import requests
        # def call_lm_arena(prompt: str, model: str = "gpt-4") -> str:
        #     """LMSYS / LM Arena API integration placeholder.
        #     Endpoint: https://arena.lmsys.org/api/v1/chat/completions
        #     Replace st.secrets['LMSYS_API_KEY'] with the real key in production."""
        #     resp = requests.post(
        #         "https://arena.lmsys.org/api/v1/chat/completions",
        #         json={"model": model,
        #               "messages": [{"role": "user", "content": prompt}]},
        #         headers={"Authorization": f"Bearer {st.secrets['LMSYS_API_KEY']}"},
        #     )
        #     return resp.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────
# TEACHER PORTAL
# ─────────────────────────────────────────────────────────────
def render_teacher(df):
    st.markdown("# 📊 Teacher Portal")
    st.markdown("*Class analytics · At-risk monitoring · SDG 4 progress*")
    st.divider()

    weak_df   = df[df["Performance"] == "Weak"]
    avg_df    = df[df["Performance"] == "Average"]
    strong_df = df[df["Performance"] == "Strong"]

    # ── KPI row ───────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Total Students",  f"{len(df):,}")
    k2.metric("🏆 Strong",          f"{len(strong_df):,}",
              f"{100*len(strong_df)/len(df):.1f}%")
    k3.metric("📈 Average",         f"{len(avg_df):,}",
              f"{100*len(avg_df)/len(df):.1f}%")
    k4.metric("⚠️ At-Risk",         f"{len(weak_df):,}",
              f"{100*len(weak_df)/len(df):.1f}%", delta_color="inverse")
    k5.metric("📐 Class Average",   f"{df['Exam_Score'].mean():.1f}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Class Overview",
        "⚠️ At-Risk Students",
        "🏆 Top Performers",
        "🌍 SDG 4 Progress",
    ])

    # ── Tab 1: Class overview ─────────────────────────────────
    with tab1:
        col_l, col_r = st.columns(2)

        with col_l:
            cat_counts = df["Performance"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_donut = px.pie(
                cat_counts, values="Count", names="Category",
                hole=0.45, title="Performance Breakdown",
                color="Category",
                color_discrete_map={"Strong": "#28a745",
                                    "Average": "#ffc107",
                                    "Weak": "#dc3545"},
            )
            fig_donut.update_layout(height=340)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_r:
            if "Gender" in df.columns:
                fig_box = px.box(df, x="Gender", y="Exam_Score", color="Gender",
                                 title="Score Distribution by Gender",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig_box.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

        # Study hours vs score trend
        df2 = df.copy()
        df2["Study_Bucket"] = pd.cut(df2["Hours_Studied"],
                                      bins=[0, 5, 10, 15, 20, 25, 30, 99],
                                      labels=["0-5","6-10","11-15","16-20",
                                              "21-25","26-30","30+"])
        bucket = df2.groupby("Study_Bucket", observed=True)["Exam_Score"].mean().reset_index()
        bucket.columns = ["Weekly Study Hours", "Average Score"]
        fig_trend = px.line(bucket, x="Weekly Study Hours", y="Average Score",
                            markers=True, title="Average Score vs Weekly Study Hours",
                            color_discrete_sequence=["royalblue"])
        fig_trend.update_layout(height=300)
        st.plotly_chart(fig_trend, use_container_width=True)

        if "School_Type" in df.columns:
            school_avg = df.groupby("School_Type")["Exam_Score"].mean().reset_index()
            school_avg.columns = ["School Type", "Average Score"]
            fig_school = px.bar(school_avg, x="School Type", y="Average Score",
                                color="Average Score", color_continuous_scale="Blues",
                                text="Average Score", title="Average Score by School Type")
            fig_school.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_school.update_layout(height=300)
            st.plotly_chart(fig_school, use_container_width=True)

    # ── Tab 2: At-risk ────────────────────────────────────────
    with tab2:
        st.warning(
            f"**{len(weak_df):,} students ({100*len(weak_df)/len(df):.1f}%)** are scoring "
            f"below {WEAK_THRESHOLD} and are flagged as at-risk.",
            icon="⚠️",
        )

        w1, w2, w3 = st.columns(3)
        w1.metric("Avg Score (at-risk)",     f"{weak_df['Exam_Score'].mean():.1f}")
        w2.metric("Avg Study Hrs (at-risk)", f"{weak_df['Hours_Studied'].mean():.1f}",
                  f"Class: {df['Hours_Studied'].mean():.1f}")
        w3.metric("Avg Attendance (at-risk)", f"{weak_df['Attendance'].mean():.1f}%",
                  f"Class: {df['Attendance'].mean():.1f}%")

        fig_scatter = px.scatter(
            weak_df, x="Hours_Studied", y="Exam_Score",
            color="Attendance", size_max=8,
            title="At-Risk Students — Study Hours vs Score (colour = Attendance %)",
            labels={"Hours_Studied": "Hours Studied / Week"},
            color_continuous_scale="RdYlGn",
        )
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

        if "Motivation_Level" in df.columns:
            mot = weak_df["Motivation_Level"].value_counts().reset_index()
            mot.columns = ["Motivation", "Count"]
            fig_mot = px.bar(mot, x="Motivation", y="Count",
                             title="Motivation Levels — At-Risk Students",
                             color_discrete_sequence=["#dc3545"])
            fig_mot.update_layout(height=280)
            st.plotly_chart(fig_mot, use_container_width=True)

        show = [c for c in
                ["Hours_Studied", "Attendance", "Sleep_Hours",
                 "Motivation_Level", "Exam_Score"] if c in df.columns]
        st.markdown("**Bottom 20 Students by Score**")
        st.dataframe(weak_df[show].sort_values("Exam_Score").head(20),
                     use_container_width=True)

    # ── Tab 3: Top performers ─────────────────────────────────
    with tab3:
        t1, t2, t3 = st.columns(3)
        t1.metric("Total Strong Students", f"{len(strong_df):,}")
        t2.metric("Highest Score",          f"{df['Exam_Score'].max():.0f}")
        t3.metric("Avg Score (top tier)",   f"{strong_df['Exam_Score'].mean():.1f}")

        if "Parental_Involvement" in df.columns:
            par = df.groupby("Parental_Involvement")["Exam_Score"].mean().reset_index()
            par.columns = ["Parental Involvement", "Avg Score"]
            fig_par = px.bar(par, x="Parental Involvement", y="Avg Score",
                             color="Avg Score", color_continuous_scale="Greens",
                             text="Avg Score", title="Avg Score by Parental Involvement")
            fig_par.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_par.update_layout(height=300)
            st.plotly_chart(fig_par, use_container_width=True)

        show2 = [c for c in
                 ["Hours_Studied", "Attendance", "Motivation_Level",
                  "School_Type", "Exam_Score"] if c in df.columns]
        st.markdown("**Top 20 Students by Score**")
        st.dataframe(strong_df[show2].sort_values("Exam_Score", ascending=False).head(20),
                     use_container_width=True)

    # ── Tab 4: SDG 4 ──────────────────────────────────────────
    with tab4:
        st.info(
            "**SDG 4 — Quality Education:** Ensure inclusive and equitable quality education "
            "and promote lifelong learning opportunities for all.",
            icon="🌍",
        )

        sdg1, sdg2 = st.columns(2)

        with sdg1:
            if "Access_to_Resources" in df.columns:
                res = df.groupby("Access_to_Resources")["Exam_Score"].mean().reset_index()
                res.columns = ["Resource Access", "Avg Score"]
                fig_res = px.bar(res, x="Resource Access", y="Avg Score",
                                 color="Avg Score", color_continuous_scale="Greens",
                                 text="Avg Score", title="Score by Resource Access Level")
                fig_res.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_res.update_layout(height=320)
                st.plotly_chart(fig_res, use_container_width=True)

        with sdg2:
            if "Internet_Access" in df.columns:
                inet = df.groupby("Internet_Access")["Exam_Score"].mean().reset_index()
                inet.columns = ["Internet Access", "Avg Score"]
                fig_inet = px.bar(inet, x="Internet Access", y="Avg Score",
                                  color_discrete_sequence=["#17a2b8", "#dc3545"],
                                  text="Avg Score", title="Score Gap: Internet Access")
                fig_inet.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_inet.update_layout(height=320)
                st.plotly_chart(fig_inet, use_container_width=True)

        # Vision 2030/2035 KPI tracker
        st.markdown("#### Vision 2030 / 2035 Progress Tracker")
        at_risk_pct   = 100 * len(weak_df) / len(df)
        avg_study_hrs = float(df["Hours_Studied"].mean())
        avg_attend    = float(df["Attendance"].mean())

        p1, p2, p3 = st.columns(3)
        p1.metric("At-Risk Rate",
                  f"{at_risk_pct:.1f}%",
                  f"{'✅ Target met' if at_risk_pct < 15 else '⚠️ Target: <15%'}",
                  delta_color="inverse")
        p2.metric("Avg Study Hours / wk",
                  f"{avg_study_hrs:.1f}",
                  f"{'✅ Target met' if avg_study_hrs >= 22 else '⚠️ Target: 22+ hrs'}",)
        p3.metric("Avg Attendance",
                  f"{avg_attend:.1f}%",
                  f"{'✅ Target met' if avg_attend >= 90 else '⚠️ Target: 90%+'}",)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    page = st.session_state.page

    if page == "home":
        render_home()
        return

    # Load data and models (cached)
    with st.spinner("Loading AI models…"):
        df = load_data()
        lr, rf, feat_cols, df_ml, metrics = train_models(df)

    if page == "student":
        render_student(df, lr, rf, feat_cols, df_ml, metrics)
    elif page == "admin":
        render_admin(df, lr, rf, feat_cols, df_ml, metrics)
    elif page == "teacher":
        render_teacher(df)
    else:
        render_home()


if __name__ == "__main__":
    main()
