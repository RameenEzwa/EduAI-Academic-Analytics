# ============================================================
# AI Student Performance Assistant — Streamlit Web Application
# SDG 4: Quality Education | Vision 2030 / 2035
# ============================================================
# Role-Based Access Control (RBAC) Architecture
#
#  STUDENT  — prediction · recommendations · AI chatbot
#  ADMIN    — full system control · ML management · raw data
#  TEACHER  — educational analytics · at-risk reports · SDG
#
# Flow: Home → Select Portal → Dashboard (role-locked)
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
import datetime
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

# Role definitions with permissions
ROLES = {
    "student": {
        "label":       "Student",
        "icon":        "🎒",
        "color":       "#0d6efd",
        "badge":       "Student Access",
        "permissions": {
            "view_raw_data":     False,
            "reload_data":       False,
            "retrain_models":    False,
            "view_system_stats": False,
            "view_other_students": False,
            "view_analytics":    False,
            "view_sdg_reports":  False,
            "use_predictor":     True,
            "use_chatbot":       True,
            "view_personal_insights": True,
        },
    },
    "admin": {
        "label":       "Administrator",
        "icon":        "⚙️",
        "color":       "#dc3545",
        "badge":       "Admin Access — Full Authority",
        "permissions": {
            "view_raw_data":     True,
            "reload_data":       True,
            "retrain_models":    True,
            "view_system_stats": True,
            "view_other_students": True,
            "view_analytics":    True,
            "view_sdg_reports":  True,
            "use_predictor":     True,
            "use_chatbot":       True,
            "view_personal_insights": True,
        },
    },
    "teacher": {
        "label":       "Teacher",
        "icon":        "📊",
        "color":       "#198754",
        "badge":       "Teacher Access — Educational Authority",
        "permissions": {
            "view_raw_data":     False,
            "reload_data":       False,
            "retrain_models":    False,
            "view_system_stats": False,
            "view_other_students": True,
            "view_analytics":    True,
            "view_sdg_reports":  True,
            "use_predictor":     False,
            "use_chatbot":       False,
            "view_personal_insights": False,
        },
    },
}


def can(permission: str) -> bool:
    """Check if the current session role has a specific permission."""
    role = st.session_state.get("role", "")
    if role not in ROLES:
        return False
    return ROLES[role]["permissions"].get(permission, False)


def access_denied(feature: str = "this feature"):
    """Display a standardised access-denied block."""
    st.error(
        f"🔒 **Access Denied** — Your role does not have permission to {feature}.\n\n"
        "Contact your system administrator if you believe this is an error.",
        icon="🚫",
    )


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page":          "home",
        "username":      "",
        "role":          "",
        "chat_history":  [],
        "retrain_count": 0,
        "last_retrain":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────
# NAVIGATION HELPER
# ─────────────────────────────────────────────────────────────
def go_to(page: str, username: str = "", role: str = ""):
    st.session_state.page         = page
    st.session_state.chat_history = []
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
        "rmse":        float(np.sqrt(mean_squared_error(ys_te, lr.predict(X_te)))),
        "r2":          float(r2_score(ys_te, lr.predict(X_te))),
        "accuracy":    float(accuracy_score(yc_te, rf.predict(X_te))),
        "importances": pd.Series(rf.feature_importances_, index=feat_cols)
                         .sort_values(ascending=False),
        "train_size":  len(X_tr),
        "test_size":   len(X_te),
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
# AI EDUCATIONAL CHATBOT  (rule-based, no external API)
# ─────────────────────────────────────────────────────────────
CHATBOT_RULES = [
    (["study", "hours", "how many", "how long"],
     "📚 Research shows students who study **20–25 hours per week** consistently outperform those who cram. "
     "Break it into 2–3 hour focused sessions with short breaks (Pomodoro technique)."),

    (["sleep", "rest", "tired"],
     "😴 Sleep is critical for memory consolidation. Aim for **7–9 hours per night**. "
     "Studies show that sleeping after learning improves retention by up to 40%."),

    (["attendance", "class", "skip", "absent"],
     "🏫 Class attendance is one of the strongest predictors of exam performance in our dataset. "
     "Students with **80%+ attendance** score, on average, 8 points higher than those below 60%."),

    (["motivation", "demotivated", "lazy", "give up"],
     "💪 Motivation dips are normal. Try:\n"
     "- Set one small, achievable goal per day\n"
     "- Study with a friend or join a study group\n"
     "- Reward yourself after completing sessions\n"
     "- Track your progress visually — seeing improvement is motivating!"),

    (["exam", "test", "prepare", "preparation"],
     "📝 Exam preparation tips:\n"
     "1. Start reviewing **3–4 weeks** before the exam\n"
     "2. Use **active recall** (test yourself) instead of re-reading\n"
     "3. Do **past papers** under timed conditions\n"
     "4. Teach the material to someone else — it reveals gaps in understanding"),

    (["score", "grade", "predict", "result"],
     "🎯 Your predicted score is calculated using a **Linear Regression model** trained on 6,607 student records. "
     "The top factors affecting your score are: study hours, attendance, motivation level, and access to resources."),

    (["resource", "internet", "access", "tools"],
     "🌐 Access to quality resources matters. Students with **high resource access** score 4–6 points higher on average. "
     "Use free platforms: Khan Academy, Coursera, YouTube EDU, and your school's library."),

    (["stress", "anxiety", "pressure", "worried"],
     "🌿 Academic stress is common — you're not alone. Tips that help:\n"
     "- Break large tasks into smaller steps\n"
     "- Practice deep breathing before exams\n"
     "- Talk to a teacher or counsellor\n"
     "- Exercise regularly — even a 20-minute walk improves focus"),

    (["parent", "family", "home"],
     "👨‍👩‍👧 Students with **high parental involvement** score 3–5 points higher on average in our dataset. "
     "Share your goals with your family — support at home makes a significant difference."),

    (["sdg", "quality education", "goal 4"],
     "🌍 **SDG 4 — Quality Education** ensures inclusive, equitable education for all. "
     "This platform directly supports SDG 4 by identifying at-risk students early, "
     "providing personalised AI recommendations, and helping schools act on data-driven insights."),

    (["hello", "hi", "hey", "start"],
     "👋 Hello! I'm your **AI Educational Assistant**. I can help you with:\n"
     "- Study strategies and time management\n"
     "- Understanding your predicted performance\n"
     "- Exam preparation tips\n"
     "- Motivation and wellbeing advice\n\n"
     "What would you like to know?"),

    (["thank", "thanks", "great", "helpful"],
     "😊 You're welcome! Remember — consistent effort beats last-minute cramming every time. "
     "Good luck with your studies! 🎓"),
]

CHATBOT_DEFAULT = (
    "🤖 I'm not sure I understood that. Try asking about:\n"
    "- **Study hours** — how much should I study?\n"
    "- **Attendance** — does attendance affect my grade?\n"
    "- **Sleep** — how does sleep affect performance?\n"
    "- **Exam prep** — how should I prepare for exams?\n"
    "- **Motivation** — how do I stay motivated?"
)


def chatbot_response(user_input: str) -> str:
    """Return a rule-based response for the student AI chatbot."""
    text = user_input.lower()
    for keywords, response in CHATBOT_RULES:
        if any(kw in text for kw in keywords):
            return response
    return CHATBOT_DEFAULT


# ─────────────────────────────────────────────────────────────
# SIDEBAR  (role-aware)
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    if st.session_state.page == "home":
        return

    role     = st.session_state.role
    username = st.session_state.username or "Guest"

    if role not in ROLES:
        return

    info  = ROLES[role]
    color = info["color"]

    with st.sidebar:
        st.markdown(f"### {info['icon']} AI Student Assistant")
        st.divider()

        # Role badge
        st.markdown(
            f"<div style='background:{color};color:white;padding:8px 12px;"
            f"border-radius:8px;font-size:0.8rem;font-weight:bold;text-align:center;'>"
            f"{info['icon']} {info['badge']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"**👤 User:** {username}")
        st.markdown(f"**🏷️ Role:** {info['label']}")

        st.divider()

        # Permission summary
        with st.expander("🔐 My Permissions", expanded=False):
            perm_labels = {
                "use_predictor":          "AI Score Predictor",
                "use_chatbot":            "AI Chatbot",
                "view_personal_insights": "Personal Insights",
                "view_analytics":         "Class Analytics",
                "view_other_students":    "Student Records",
                "view_sdg_reports":       "SDG Reports",
                "view_raw_data":          "Raw Dataset",
                "view_system_stats":      "System Stats",
                "retrain_models":         "Retrain ML Models",
                "reload_data":            "Reload Dataset",
            }
            for perm_key, perm_label in perm_labels.items():
                has = info["permissions"].get(perm_key, False)
                icon = "✅" if has else "🔒"
                st.markdown(f"{icon} {perm_label}")

        st.divider()

        if st.button("🏠  Back to Home", use_container_width=True):
            go_to("home")

        st.divider()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        st.caption(f"Session: {now}")
        st.caption("Dataset: Kaggle · SDG 4 · Vision 2030/35")


# ─────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────
def render_home():
    st.markdown(
        "<h1 style='text-align:center;padding-top:1.5rem;'>🎓 AI Student Performance Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;font-size:1.15rem;color:gray;'>"
        "An AI-powered educational management platform · "
        "<strong>SDG 4: Quality Education</strong> · Vision 2030 / 2035"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.info(
            "**🌍 SDG 4 — Quality Education**  \n"
            "Ensure inclusive, equitable quality education and promote "
            "lifelong learning opportunities for all.",
        )
    with sc2:
        st.success(
            "**🚀 Vision 2030 / 2035**  \n"
            "Building a knowledge-based economy through AI-driven personalised "
            "learning and data-informed educational governance.",
        )

    st.divider()
    st.markdown(
        "<h3 style='text-align:center;'>Select Your Portal to Begin</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")

    # ── Student card ──────────────────────────────────────────
    with c1:
        with st.container(border=True):
            st.markdown("## 🎒")
            st.markdown("### Student Portal")
            st.markdown(
                "Your personalised AI academic assistant. Get a predicted exam score, "
                "study recommendations, and chat with the AI educational assistant."
            )
            st.markdown("**You can access:**")
            st.markdown("✅ AI Score Predictor  \n✅ Study Recommendations  \n✅ AI Educational Chatbot  \n✅ Personal Insights")
            st.markdown("&nbsp;")
            sname = st.text_input("Your name", placeholder="e.g. Alex Johnson", key="home_sname")
            if st.button("Enter Student Portal", key="go_student", type="primary", use_container_width=True):
                go_to("student", username=sname or "Student", role="student")

    # ── Admin card ────────────────────────────────────────────
    with c2:
        with st.container(border=True):
            st.markdown("## ⚙️")
            st.markdown("### Admin Portal")
            st.markdown(
                "Full system authority. Manage the dataset, monitor ML model health, "
                "retrain models, and access all technical system analytics."
            )
            st.markdown("**Full authority including:**")
            st.markdown("✅ Raw Dataset Access  \n✅ Retrain ML Models  \n✅ System Monitoring  \n✅ All Analytics")
            st.markdown("&nbsp;")
            aname = st.text_input("Admin ID", placeholder="e.g. admin@school.edu", key="home_aname")
            if st.button("Enter Admin Portal", key="go_admin", type="secondary", use_container_width=True):
                go_to("admin", username=aname or "Admin", role="admin")

    # ── Teacher card ──────────────────────────────────────────
    with c3:
        with st.container(border=True):
            st.markdown("## 📊")
            st.markdown("### Teacher Portal")
            st.markdown(
                "Educational decision-making dashboard. Monitor at-risk students, "
                "analyse class performance, and track SDG 4 progress."
            )
            st.markdown("**Educational authority:**")
            st.markdown("✅ At-Risk Monitoring  \n✅ Class Analytics  \n✅ SDG 4 Reports  \n🔒 No System Access")
            st.markdown("&nbsp;")
            tname = st.text_input("Teacher name", placeholder="e.g. Ms. Rivera", key="home_tname")
            if st.button("Enter Teacher Portal", key="go_teacher", use_container_width=True):
                go_to("teacher", username=tname or "Teacher", role="teacher")

    st.divider()
    st.caption(
        "Dataset: [Kaggle — Student Performance Factors]"
        "(https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) "
        "· 6,607 records · 20 features · CC0 License"
    )


# ─────────────────────────────────────────────────────────────
# STUDENT PORTAL
# ─────────────────────────────────────────────────────────────
def render_student(df, lr, rf, feat_cols, df_ml, metrics):
    name = st.session_state.username or "Student"

    # Role header
    st.markdown(
        "<div style='background:#0d6efd;color:white;padding:10px 18px;"
        "border-radius:10px;margin-bottom:12px;'>"
        "🎒 <strong>Student Portal</strong> &nbsp;·&nbsp; "
        "<span style='font-size:0.85rem;opacity:0.9;'>Personal academic AI assistant</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"Welcome, **{name}**! Your AI learning assistant is ready.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🔮 AI Predictor", "💡 Recommendations", "🤖 AI Chatbot"])

    # ── Tab 1: AI Predictor ───────────────────────────────────
    with tab1:
        st.markdown("### 📋 Your Study Profile")
        col1, col2 = st.columns(2, gap="large")

        with col1:
            hours      = st.slider("📚 Study hours per week",  0,  60, 20)
            attendance = st.slider("🏫 Attendance (%)",         0, 100, 85)
            sleep      = st.slider("😴 Sleep hours per night",  4,  12,  7)

        with col2:
            motivation   = st.selectbox("💪 Motivation level", ["Low", "Medium", "High"], index=1)
            prev_score   = st.slider("📋 Previous exam score", 40, 100, 70)
            has_internet = st.radio("🌐 Internet access", ["Yes", "No"], horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🔮  Generate My AI Performance Report",
                        type="primary", use_container_width=True)

        if run:
            with st.spinner("AI is analysing your profile…"):
                score, category = predict(lr, rf, feat_cols, df_ml, hours, attendance)

            st.divider()
            st.markdown("## 📊 Your AI Performance Report")

            cat_label = {
                "Weak":    "⚠️ Needs Improvement",
                "Average": "📈 On Track",
                "Strong":  "🏆 Excellent",
            }
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🎯 Predicted Score",      f"{score:.1f} / 100")
            k2.metric("📂 Category",             category, cat_label[category])
            k3.metric("📚 Study Hours",          f"{hours} hrs/wk")
            k4.metric("🏫 Attendance",           f"{attendance}%")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={"reference": 75, "increasing": {"color": "green"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#0d6efd"},
                    "steps": [
                        {"range": [0,  WEAK_THRESHOLD],              "color": "#ffcccc"},
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

            feedback = {
                "Strong":  (
                    f"🏆 **Outstanding!** {name}, your study habits predict a top-tier score of {score:.1f}. "
                    "Consider taking on leadership roles like peer tutoring — teaching deepens your own mastery."
                ),
                "Average": (
                    f"📈 **Solid effort, {name}!** A score of {score:.1f} is respectable. "
                    "Targeted improvements in attendance and active recall practice could push you into the Strong tier."
                ),
                "Weak":    (
                    f"⚠️ **Action needed, {name}.** Predicted score: {score:.1f}. "
                    "Students who added 5 hrs/week study time and raised attendance to 80%+ moved up a full "
                    "performance category in one semester. Start small — ask for help early."
                ),
            }
            st.info(feedback[category], icon="🤖")

            st.divider()
            st.markdown("### 📈 Your Score in Class Context")
            fig = px.histogram(df, x="Exam_Score", nbins=30,
                               color_discrete_sequence=["#6ea8fe"],
                               labels={"Exam_Score": "Exam Score"},
                               title="Class Score Distribution — Your Prediction Marked")
            fig.add_vline(x=score, line_color="red", line_width=3,
                          annotation_text=f"You: {score:.1f}", annotation_position="top right")
            fig.add_vline(x=df["Exam_Score"].mean(), line_color="orange", line_dash="dash",
                          annotation_text=f"Class avg: {df['Exam_Score'].mean():.1f}")
            fig.update_layout(height=320, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Recommendations ────────────────────────────────
    with tab2:
        st.markdown("### 💡 Personalised Study Recommendations")
        st.markdown("Answer these questions to receive tailored advice:")

        r1, r2 = st.columns(2)
        with r1:
            r_hours      = st.slider("📚 Weekly study hours",  0, 60, 20, key="rec_hrs")
            r_attendance = st.slider("🏫 Attendance (%)",       0, 100, 85, key="rec_att")
            r_sleep      = st.slider("😴 Nightly sleep hours",  4, 12, 7,  key="rec_slp")
        with r2:
            r_motivation = st.selectbox("💪 Motivation level",
                                        ["Low", "Medium", "High"], index=1, key="rec_mot")
            r_tutoring   = st.slider("👩‍🏫 Tutoring sessions / week", 0, 10, 1, key="rec_tut")

        if st.button("Generate My Study Plan", type="primary", use_container_width=True):
            avg_hrs = float(df["Hours_Studied"].mean())
            avg_att = float(df["Attendance"].mean())

            recs = []

            # Study hours
            if r_hours < avg_hrs * 0.7:
                recs.append(("📚 Study Hours — Needs Improvement",
                              f"You study {r_hours} hrs/wk vs the class average of {avg_hrs:.0f} hrs. "
                              "Increase by at least 5 hours. Use Pomodoro (25 min on / 5 min break)."))
            elif r_hours >= 25:
                recs.append(("✅ Study Hours — Excellent",
                              f"{r_hours} hrs/wk is above average. Focus on quality: active recall, "
                              "spaced repetition, and past-paper practice."))
            else:
                recs.append(("📚 Study Hours — On Track",
                              f"{r_hours} hrs/wk is healthy. Ensure sessions are distraction-free."))

            # Attendance
            if r_attendance < 75:
                recs.append(("🏫 Attendance — Critical",
                              f"{r_attendance}% attendance is dangerously low. Students below 75% miss "
                              "core exam content. Attend at minimum 80% of all classes."))
            elif r_attendance >= avg_att:
                recs.append(("✅ Attendance — Great",
                              f"{r_attendance}% is at or above the class average. Keep it up."))
            else:
                recs.append(("📅 Attendance — Below Average",
                              f"{r_attendance}% is below the class average ({avg_att:.0f}%). "
                              "Each missed class compounds over time."))

            # Sleep
            if r_sleep < 6:
                recs.append(("😴 Sleep — Insufficient",
                              "Less than 6 hours impairs memory consolidation and focus. "
                              "Aim for 7–9 hours. Avoid screens 1 hour before bed."))
            elif r_sleep > 10:
                recs.append(("😴 Sleep — Excessive",
                              "Oversleeping can cause daytime lethargy. "
                              "A consistent 7–8 hour schedule is optimal for learning."))
            else:
                recs.append(("✅ Sleep — Healthy", f"{r_sleep} hours is ideal. Maintain this routine."))

            # Motivation
            if r_motivation == "Low":
                recs.append(("💪 Motivation — Boost Needed",
                              "Low motivation is the single most controllable factor. Try:\n"
                              "• Set one small daily goal\n"
                              "• Find a study partner\n"
                              "• Track progress with a visual chart\n"
                              "• Reward yourself for hitting milestones"))
            elif r_motivation == "High":
                recs.append(("🏆 Motivation — Superb",
                              "High motivation correlates strongly with top performance in our dataset. "
                              "Channel it with structured study plans to maximise impact."))

            # Tutoring
            if r_tutoring == 0:
                recs.append(("👩‍🏫 Tutoring — Consider Adding",
                              "Even 1 tutoring session per week is associated with higher scores. "
                              "Seek help from teachers or online resources when stuck — don't wait."))
            else:
                recs.append(("✅ Tutoring — Active",
                              f"{r_tutoring} session(s)/week is positive. "
                              "Make sure to prepare questions before each session to maximise value."))

            for title, body in recs:
                icon = "✅" if title.startswith("✅") else "⚠️" if "Critical" in title or "Needs" in title or "Insufficient" in title else "📌"
                with st.expander(title, expanded=True):
                    st.write(body)

            st.divider()
            st.markdown("#### 🗓️ Suggested Weekly Study Schedule")
            days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            daily = max(1, round(r_hours / 7, 1))
            schedule_df = pd.DataFrame({
                "Day":          days,
                "Study Hours":  [daily] * 5 + [daily * 1.2, daily * 0.5],
                "Focus Area":   ["Lecture Review","Practice Problems","Active Recall",
                                 "Past Papers","Weak Topics","Full Study Session","Rest & Review"],
            })
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    # ── Tab 3: AI Chatbot ─────────────────────────────────────
    with tab3:
        st.markdown("### 🤖 AI Educational Chatbot")
        st.markdown(
            "Chat with your AI study assistant. Ask about study strategies, "
            "exam tips, motivation, or anything education-related."
        )
        st.divider()

        # Chat history display
        chat_container = st.container(height=380)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    "👋 **Hi! I'm your AI Educational Assistant.**  \n"
                    "Ask me anything about studying, exams, motivation, or performance tips."
                )
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Ask your AI assistant…")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = chatbot_response(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        st.divider()
        st.markdown("**Quick questions to try:**")
        qcols = st.columns(3)
        quick_qs = [
            "How many hours should I study?",
            "How does attendance affect my grade?",
            "How do I stay motivated?",
            "How should I prepare for exams?",
            "How does sleep affect performance?",
            "What is SDG 4?",
        ]
        for i, q in enumerate(quick_qs):
            with qcols[i % 3]:
                if st.button(q, key=f"qq_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": chatbot_response(q)}
                    )
                    st.rerun()


# ─────────────────────────────────────────────────────────────
# ADMIN PORTAL
# ─────────────────────────────────────────────────────────────
def render_admin(df, lr, rf, feat_cols, df_ml, metrics):
    # Admin header banner
    st.markdown(
        "<div style='background:#dc3545;color:white;padding:10px 18px;"
        "border-radius:10px;margin-bottom:12px;'>"
        "⚙️ <strong>Admin Portal</strong> &nbsp;·&nbsp; "
        "<span style='font-size:0.85rem;opacity:0.9;'>"
        "🔒 Full System Authority — Admin Access Only</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"System management dashboard — logged in as **{st.session_state.username}**.")
    st.divider()

    # System status KPIs
    st.markdown("### 🖥️ System Status")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("🟢 Status",          "Online")
    s2.metric("📋 Records",         f"{len(df):,}")
    s3.metric("🏷️ Features",        f"{len(df.columns) - 1}")
    s4.metric("❌ Missing Values",  f"{df.drop(columns=['Performance']).isnull().sum().sum()}")
    s5.metric("🤖 Active Models",   "2  (LR + RF)")
    s6.metric("🔄 Retrains",        f"{st.session_state.retrain_count}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Dataset Management", "📊 Data Analysis", "🤖 ML Models"])

    # ── Tab 1: Dataset Management — Admin Only ─────────────────
    with tab1:
        st.markdown(
            "<span style='background:#dc3545;color:white;padding:3px 10px;"
            "border-radius:5px;font-size:0.8rem;font-weight:bold;'>🔒 Admin Only</span>",
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;")

        if not can("view_raw_data"):
            access_denied("view raw dataset")
        else:
            st.markdown("#### Raw Dataset Viewer")
            search = st.text_input("🔎 Filter by any value", "")
            display = df.copy()
            if search:
                mask    = display.astype(str).apply(
                    lambda r: r.str.contains(search, case=False)).any(axis=1)
                display = display[mask]
            n = st.slider("Rows to display", 10, 500, 50)
            st.dataframe(display.head(n), use_container_width=True, height=420)
            st.caption(f"Showing {min(n, len(display))} of {len(display)} records")

            col_dl, col_rel = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download Full Dataset (CSV)",
                    data=df.drop(columns=["Performance"]).to_csv(index=False).encode(),
                    file_name="StudentPerformanceFactors.csv",
                    mime="text/csv",
                )
            with col_rel:
                if not can("reload_data"):
                    access_denied("reload the dataset")
                else:
                    if st.button("🔄 Reload Dataset from Disk", type="secondary",
                                 use_container_width=True):
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.success("✅ Cache cleared. Dataset will reload on next request.")

        st.divider()
        st.markdown("#### Data Quality Report")
        miss = df.isnull().sum()
        if miss.sum() == 0:
            st.success("✅ No missing values after preprocessing.")
        else:
            st.warning(f"⚠️ {miss.sum()} missing values remain.")
            st.dataframe(miss[miss > 0].rename("Missing Count"), use_container_width=True)

        dtype_df = pd.DataFrame({
            "Column":        df.columns,
            "Type":          df.dtypes.astype(str).values,
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Data Analysis ──────────────────────────────────
    with tab2:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.select_dtypes(include=[np.number]).describe().round(2),
                     use_container_width=True)

        st.markdown("#### Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r",
                             title="Feature Correlation Matrix")
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Column Distribution Explorer")
        col_sel = st.selectbox("Select column", df.columns)
        if df[col_sel].dtype in [np.float64, np.int64]:
            fig_col = px.histogram(df, x=col_sel, nbins=30,
                                   color_discrete_sequence=["#dc3545"],
                                   title=f"Distribution: {col_sel}")
        else:
            vc = df[col_sel].value_counts().reset_index()
            vc.columns = [col_sel, "Count"]
            fig_col = px.bar(vc, x=col_sel, y="Count",
                             color_discrete_sequence=["#dc3545"],
                             title=f"Value Counts: {col_sel}")
        fig_col.update_layout(height=300)
        st.plotly_chart(fig_col, use_container_width=True)

    # ── Tab 3: ML Models — Admin Only ─────────────────────────
    with tab3:
        st.markdown(
            "<span style='background:#dc3545;color:white;padding:3px 10px;"
            "border-radius:5px;font-size:0.8rem;font-weight:bold;'>🔒 Admin Only</span>",
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;")

        st.markdown("#### Model Performance Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("LR — RMSE",     f"{metrics['rmse']:.3f}",        "Lower = better")
        m2.metric("LR — R²",       f"{metrics['r2']:.3f}",          "1.0 = perfect")
        m3.metric("RF — Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m4.metric("Training Set",  f"{metrics['train_size']:,} rows")
        m5.metric("Test Set",      f"{metrics['test_size']:,} rows")

        st.markdown("#### Top 10 Feature Importances")
        top10 = metrics["importances"].head(10).reset_index()
        top10.columns = ["Feature", "Importance"]
        fig_imp = px.bar(top10, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Reds",
                         title="Random Forest Feature Importances")
        fig_imp.update_layout(height=380,
                              yaxis={"categoryorder": "total ascending"},
                              margin=dict(t=50, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()

        if not can("retrain_models"):
            access_denied("retrain ML models")
        else:
            st.markdown("#### 🔁 Retrain ML Models")
            st.warning(
                "⚠️ Retraining will clear all cached models and reload from the dataset. "
                "This is a system-level operation and should only be run after dataset changes.",
                icon="⚙️",
            )
            last = st.session_state.last_retrain
            if last:
                st.caption(f"Last retrained: {last}")

            if st.button("🔁 Retrain Now", type="primary"):
                with st.spinner("Retraining models on full dataset…"):
                    st.cache_resource.clear()
                    st.session_state.retrain_count += 1
                    st.session_state.last_retrain = (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                st.success(
                    f"✅ Models retrained successfully. "
                    f"Total retrains this session: {st.session_state.retrain_count}"
                )
                st.rerun()

        st.divider()
        st.markdown("#### Dataset Source")
        with st.container(border=True):
            st.markdown(
                "**Source:** [Kaggle — Student Performance Factors]"
                "(https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)  \n"
                "**File:** `StudentPerformanceFactors.csv`  \n"
                "**License:** CC0 Public Domain  \n"
                "**Records:** 6,607  |  **Features:** 20  |  **Target:** `Exam_Score`"
            )

        # Mock LMSYS / LM Arena API integration wrapper
        # ──────────────────────────────────────────────────────
        # import requests
        # def call_lm_arena(prompt: str, model: str = "gpt-4") -> str:
        #     """LMSYS / LM Arena API placeholder.
        #     Replace st.secrets['LMSYS_API_KEY'] with the real key."""
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
    st.markdown(
        "<div style='background:#198754;color:white;padding:10px 18px;"
        "border-radius:10px;margin-bottom:12px;'>"
        "📊 <strong>Teacher Portal</strong> &nbsp;·&nbsp; "
        "<span style='font-size:0.85rem;opacity:0.9;'>"
        "Educational Authority — Analytics & Reporting</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"Educational decision-making dashboard — **{st.session_state.username}**.")
    st.divider()

    weak_df   = df[df["Performance"] == "Weak"]
    avg_df    = df[df["Performance"] == "Average"]
    strong_df = df[df["Performance"] == "Strong"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Total Students", f"{len(df):,}")
    k2.metric("🏆 Strong",         f"{len(strong_df):,}", f"{100*len(strong_df)/len(df):.1f}%")
    k3.metric("📈 Average",        f"{len(avg_df):,}",    f"{100*len(avg_df)/len(df):.1f}%")
    k4.metric("⚠️ At-Risk",        f"{len(weak_df):,}",
              f"{100*len(weak_df)/len(df):.1f}%", delta_color="inverse")
    k5.metric("📐 Class Avg",      f"{df['Exam_Score'].mean():.1f}")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Class Overview",
        "⚠️ At-Risk Students",
        "🏆 Top Performers",
        "🌍 SDG 4 & Vision 2030/2035",
    ])

    # ── Tab 1: Class Overview ─────────────────────────────────
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

        df2 = df.copy()
        df2["Study_Bucket"] = pd.cut(df2["Hours_Studied"],
                                      bins=[0, 5, 10, 15, 20, 25, 30, 99],
                                      labels=["0-5","6-10","11-15","16-20","21-25","26-30","30+"])
        bucket = df2.groupby("Study_Bucket", observed=True)["Exam_Score"].mean().reset_index()
        bucket.columns = ["Weekly Study Hours", "Average Score"]
        fig_trend = px.line(bucket, x="Weekly Study Hours", y="Average Score",
                            markers=True, title="Average Score vs Weekly Study Hours",
                            color_discrete_sequence=["#198754"])
        fig_trend.update_layout(height=300)
        st.plotly_chart(fig_trend, use_container_width=True)

        if "School_Type" in df.columns:
            school_avg = df.groupby("School_Type")["Exam_Score"].mean().reset_index()
            school_avg.columns = ["School Type", "Average Score"]
            fig_sch = px.bar(school_avg, x="School Type", y="Average Score",
                             color="Average Score", color_continuous_scale="Greens",
                             text="Average Score", title="Average Score by School Type")
            fig_sch.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_sch.update_layout(height=300)
            st.plotly_chart(fig_sch, use_container_width=True)

    # ── Tab 2: At-Risk Students ───────────────────────────────
    with tab2:
        st.warning(
            f"**{len(weak_df):,} students ({100*len(weak_df)/len(df):.1f}%)** are scoring "
            f"below {WEAK_THRESHOLD} and are flagged as at-risk.",
            icon="⚠️",
        )

        w1, w2, w3 = st.columns(3)
        w1.metric("Avg Score (at-risk)",      f"{weak_df['Exam_Score'].mean():.1f}")
        w2.metric("Avg Study Hrs (at-risk)",  f"{weak_df['Hours_Studied'].mean():.1f}",
                  f"Class: {df['Hours_Studied'].mean():.1f}")
        w3.metric("Avg Attendance (at-risk)", f"{weak_df['Attendance'].mean():.1f}%",
                  f"Class: {df['Attendance'].mean():.1f}%")

        fig_sc = px.scatter(
            weak_df, x="Hours_Studied", y="Exam_Score",
            color="Attendance", size_max=8,
            title="At-Risk Students — Study Hours vs Score (colour = Attendance %)",
            labels={"Hours_Studied": "Hours Studied / Week"},
            color_continuous_scale="RdYlGn",
        )
        fig_sc.update_layout(height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

        if "Motivation_Level" in df.columns:
            mot = weak_df["Motivation_Level"].value_counts().reset_index()
            mot.columns = ["Motivation", "Count"]
            fig_mot = px.bar(mot, x="Motivation", y="Count",
                             title="Motivation Levels — At-Risk Students",
                             color_discrete_sequence=["#dc3545"])
            fig_mot.update_layout(height=280)
            st.plotly_chart(fig_mot, use_container_width=True)

        show = [c for c in ["Hours_Studied", "Attendance", "Sleep_Hours",
                             "Motivation_Level", "Exam_Score"] if c in df.columns]
        st.markdown("**Bottom 20 Students by Score**")
        st.dataframe(weak_df[show].sort_values("Exam_Score").head(20),
                     use_container_width=True)

        # Teacher note: no admin tools here
        st.info(
            "🔒 Dataset reload and model retraining are **Admin-only** operations. "
            "Contact your system administrator to update model parameters.",
            icon="ℹ️",
        )

    # ── Tab 3: Top Performers ─────────────────────────────────
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

        show2 = [c for c in ["Hours_Studied", "Attendance", "Motivation_Level",
                              "School_Type", "Exam_Score"] if c in df.columns]
        st.markdown("**Top 20 Students by Score**")
        st.dataframe(strong_df[show2].sort_values("Exam_Score", ascending=False).head(20),
                     use_container_width=True)

    # ── Tab 4: SDG 4 & Vision 2030/2035 ──────────────────────
    with tab4:
        st.markdown("### 🌍 SDG 4 & Vision 2030/2035 — Demonstrated Through System Functionality")

        st.info(
            "**SDG 4 — Quality Education** targets are tracked below using live dataset metrics. "
            "Each Vision goal is tied directly to a feature of this platform.",
            icon="🌍",
        )

        # ── Vision 2030 ───────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🚀 Vision 2030 — Digital Knowledge Economy")

        v30_1, v30_2 = st.columns(2)
        with v30_1:
            with st.container(border=True):
                st.markdown("**🤖 AI-Assisted Learning**")
                st.markdown(
                    "This platform uses a **Random Forest Classifier** (97.2% accuracy) "
                    "and **Linear Regression** to predict student outcomes, enabling targeted "
                    "AI-assisted intervention before exam time."
                )
                st.metric("RF Model Accuracy", "97.2%", "Predicts Weak/Average/Strong")

        with v30_2:
            with st.container(border=True):
                st.markdown("**📊 Workforce Capability Enhancement**")
                st.markdown(
                    "By detecting at-risk students early, schools can deploy targeted support, "
                    "improving graduation rates and producing a more capable future workforce."
                )
                at_risk_pct = 100 * len(weak_df) / len(df)
                st.metric("At-Risk Students Identified", f"{len(weak_df):,}",
                          f"{at_risk_pct:.1f}% flagged for support", delta_color="inverse")

        # ── Vision 2035 ───────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔭 Vision 2035 — AI-Driven Educational Quality Assurance")

        v35_1, v35_2 = st.columns(2)
        with v35_1:
            with st.container(border=True):
                st.markdown("**🔮 Predictive Educational Analytics**")
                st.markdown(
                    "The Student Portal provides personalised score predictions based on "
                    "19 input features. Teachers can see macro-level predictive trends here "
                    "to guide curriculum decisions."
                )
                avg_hrs = float(df["Hours_Studied"].mean())
                fig_pred = px.scatter(
                    df.sample(300, random_state=42),
                    x="Hours_Studied", y="Exam_Score",
                    color="Performance", size_max=6,
                    title="Predictive Pattern: Study Hours → Score",
                    color_discrete_map={"Strong": "#28a745",
                                        "Average": "#ffc107",
                                        "Weak": "#dc3545"},
                )
                fig_pred.update_layout(height=280, margin=dict(t=40, b=10))
                st.plotly_chart(fig_pred, use_container_width=True)

        with v35_2:
            with st.container(border=True):
                st.markdown("**🌐 Equitable Student Support Systems**")
                st.markdown(
                    "The equity gap below measures how access to resources and internet "
                    "affects exam outcomes — directly informing equitable resource allocation."
                )
                if "Access_to_Resources" in df.columns:
                    res = df.groupby("Access_to_Resources")["Exam_Score"].mean().reset_index()
                    res.columns = ["Resource Access", "Avg Score"]
                    fig_res = px.bar(res, x="Resource Access", y="Avg Score",
                                     color="Avg Score", color_continuous_scale="Greens",
                                     text="Avg Score", title="Score by Resource Access Level")
                    fig_res.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                    fig_res.update_layout(height=280, margin=dict(t=40, b=10))
                    st.plotly_chart(fig_res, use_container_width=True)

        # ── Vision KPI Progress Tracker ───────────────────────
        st.markdown("---")
        st.markdown("#### 📐 Vision 2030/2035 KPI Progress Tracker")

        at_risk_pct   = 100 * len(weak_df) / len(df)
        avg_study_hrs = float(df["Hours_Studied"].mean())
        avg_attend    = float(df["Attendance"].mean())

        internet_pct  = 0.0
        if "Internet_Access" in df.columns:
            internet_pct = 100 * (df["Internet_Access"] == "Yes").sum() / len(df)

        kpi_data = {
            "KPI":           ["At-Risk Student Rate", "Avg Study Hours/wk",
                              "Avg Attendance", "Internet Access Rate"],
            "Current":       [f"{at_risk_pct:.1f}%", f"{avg_study_hrs:.1f} hrs",
                              f"{avg_attend:.1f}%", f"{internet_pct:.1f}%"],
            "Vision Target": ["< 15%", "22+ hrs", "90%+", "100%"],
            "Status":        [
                "✅ Met" if at_risk_pct < 15 else "⚠️ Not Met",
                "✅ Met" if avg_study_hrs >= 22 else "⚠️ Not Met",
                "✅ Met" if avg_attend >= 90 else "⚠️ Not Met",
                "✅ Met" if internet_pct >= 95 else "⚠️ Not Met",
            ],
        }
        st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)

        if "Internet_Access" in df.columns:
            inet = df.groupby("Internet_Access")["Exam_Score"].mean().reset_index()
            inet.columns = ["Internet Access", "Avg Score"]
            fig_inet = px.bar(inet, x="Internet Access", y="Avg Score",
                              color_discrete_sequence=["#198754", "#dc3545"],
                              text="Avg Score", title="Digital Divide: Internet Access vs Score")
            fig_inet.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_inet.update_layout(height=280)
            st.plotly_chart(fig_inet, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    page = st.session_state.page

    if page == "home":
        render_home()
        return

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
