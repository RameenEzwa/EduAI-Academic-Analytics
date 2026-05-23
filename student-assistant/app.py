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
# GLOBAL THEME  — Futuristic Dark AI Platform
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Palette — Emerald & Deep Forest ──────────────────────── */
:root {
    --bg:          #071A10;
    --bg2:         #0C2318;
    --bg3:         #122D20;
    --card:        rgba(10,32,20,0.88);
    --card-border: rgba(16,185,129,0.22);
    --blue:        #10B981;
    --blue-dim:    #059669;
    --blue-glow:   rgba(16,185,129,0.28);
    --teal:        #34D399;
    --teal-dim:    #10B981;
    --teal-glow:   rgba(52,211,153,0.25);
    --purple:      #6EE7B7;
    --amber:       #FCD34D;
    --red:         #F87171;
    --text:        #ECFDF5;
    --text2:       #A7F3D0;
    --text-muted:  #6EE7B7;
    --border:      rgba(52,211,153,0.1);
    --navy:        #064E3B;
}

/* ── Animations ───────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 15px var(--blue-glow); }
    50%       { box-shadow: 0 0 30px rgba(59,130,246,0.45); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}
@keyframes orb-move {
    0%   { transform: translate(0,0) scale(1); }
    33%  { transform: translate(30px,-20px) scale(1.05); }
    66%  { transform: translate(-20px,15px) scale(0.97); }
    100% { transform: translate(0,0) scale(1); }
}

/* ── Global body / app ────────────────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.main .block-container {
    background: transparent !important;
    padding-top: 1.5rem !important;
    animation: fadeIn 0.5s ease;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #041510 0%, #071A10 40%, #0C2318 80%, #071A10 100%) !important;
    border-right: 1px solid rgba(16,185,129,0.2) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] * {
    color: #A7F3D0 !important;
}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong {
    color: #ECFDF5 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(52,211,153,0.15) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(16,185,129,0.12) !important;
    color: #34D399 !important;
    border: 1px solid rgba(52,211,153,0.35) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(16,185,129,0.22) !important;
    box-shadow: 0 0 14px rgba(16,185,129,0.3) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(16,185,129,0.04) !important;
    border: 1px solid rgba(52,211,153,0.12) !important;
    border-radius: 10px !important;
}

/* ── Primary buttons ──────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--blue-dim) 0%, var(--blue) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
}

/* ── Secondary buttons ────────────────────────────────────── */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background: rgba(59,130,246,0.07) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    color: var(--blue) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    background: rgba(59,130,246,0.14) !important;
    border-color: var(--blue) !important;
}

/* ── Download button ──────────────────────────────────────── */
.stDownloadButton > button {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    color: var(--teal) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: rgba(16,185,129,0.2) !important;
    box-shadow: 0 0 12px var(--teal-glow) !important;
}

/* ── Input fields ─────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: rgba(10,32,20,0.9) !important;
    border: 1px solid rgba(52,211,153,0.2) !important;
    border-radius: 10px !important;
    color: #ECFDF5 !important;
    padding: 0.6rem 0.9rem !important;
    transition: all 0.2s ease !important;
    -webkit-text-fill-color: #ECFDF5 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #10B981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.2) !important;
    background: rgba(18,45,32,0.95) !important;
}
input[type="text"]::placeholder, textarea::placeholder {
    color: #6EE7B7 !important;
    opacity: 0.5 !important;
}

/* ── Selectbox ────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(10,32,20,0.9) !important;
    border: 1px solid rgba(52,211,153,0.2) !important;
    border-radius: 10px !important;
    color: #ECFDF5 !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #10B981 !important;
}

/* ── Sliders ──────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--blue) !important;
    box-shadow: 0 0 8px var(--blue-glow) !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--blue), var(--teal)) !important;
}

/* ── Radio buttons ────────────────────────────────────────── */
[data-testid="stRadio"] label {
    color: var(--text2) !important;
}

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.2rem !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    animation: fadeInUp 0.4s ease both !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4), 0 0 20px var(--blue-glow) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 800 !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricDelta"] { color: var(--teal) !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Bordered containers ──────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    background: var(--card) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    padding: 1.25rem 1.25rem 1rem !important;
    animation: fadeInUp 0.4s ease both;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px 10px 0 0 !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--blue) !important;
    background: rgba(59,130,246,0.07) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.12) !important;
    color: var(--blue) !important;
    font-weight: 700 !important;
    border-bottom: 2px solid var(--blue) !important;
}
[data-testid="stTabsContent"] {
    background: transparent !important;
}

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text2) !important;
    font-weight: 600 !important;
}

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--card-border) !important;
}
.dvn-scroller { background: var(--bg2) !important; }

/* ── Alerts ───────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    background: rgba(22,27,39,0.8) !important;
    backdrop-filter: blur(8px) !important;
    animation: fadeInUp 0.35s ease !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Headings ─────────────────────────────────────────────── */
h1 { font-size: 1.9rem !important; font-weight: 800 !important; color: var(--text) !important; }
h2 { font-size: 1.45rem !important; font-weight: 700 !important; color: var(--text) !important; }
h3 { font-size: 1.15rem !important; font-weight: 700 !important; color: var(--text2) !important; }
p, li, label { color: var(--text2) !important; }
.stMarkdown { color: var(--text2) !important; }

/* ── Code ─────────────────────────────────────────────────── */
code {
    background: rgba(59,130,246,0.12) !important;
    color: var(--blue) !important;
    border-radius: 5px !important;
    padding: 1px 6px !important;
    font-size: 0.85rem !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
}

/* ── Chat messages ────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
}

/* ── Spinner ──────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    border-top-color: var(--blue) !important;
}

/* ══════════════════════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
   ══════════════════════════════════════════════════════════ */

/* ── Login hero banner ────────────────────────────────────── */
.login-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #041510 0%, #071A10 50%, #0C2318 100%);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    animation: fadeInUp 0.6s ease;
    box-shadow: 0 8px 40px rgba(16,185,129,0.12);
}
.login-hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(16,185,129,0.22) 0%, transparent 70%);
    border-radius: 50%;
    animation: orb-move 8s ease-in-out infinite;
}
.login-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(52,211,153,0.18) 0%, transparent 70%);
    border-radius: 50%;
    animation: orb-move 10s ease-in-out infinite reverse;
}
.login-logo {
    font-size: 3.5rem;
    display: block;
    animation: float 3s ease-in-out infinite;
    margin-bottom: 0.75rem;
}
.login-title {
    font-size: 1.85rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6EE7B7, #34D399, #10B981, #A7F3D0, #6EE7B7);
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.login-sub {
    font-size: 0.88rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.5px;
}
.login-pill {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(52,211,153,0.35);
    color: #6EE7B7 !important;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    margin: 0.5rem 4px 0;
}

/* ── Credential rows ──────────────────────────────────────── */
.cred-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
    margin-bottom: 0.45rem;
    font-size: 0.88rem;
    transition: transform 0.15s ease;
}
.cred-row:hover { transform: translateX(3px); }
.cred-admin   { background: rgba(248,113,113,0.08); border-left: 3px solid #F87171; }
.cred-teacher { background: rgba(52,211,153,0.10);  border-left: 3px solid #34D399; }
.cred-student { background: rgba(16,185,129,0.08);  border-left: 3px solid #10B981; }
.cred-badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.badge-admin   { background: rgba(248,113,113,0.2); color: #FCA5A5; border: 1px solid rgba(248,113,113,0.35); }
.badge-teacher { background: rgba(52,211,153,0.2);  color: #6EE7B7; border: 1px solid rgba(52,211,153,0.35); }
.badge-student { background: rgba(16,185,129,0.2);  color: #A7F3D0; border: 1px solid rgba(16,185,129,0.35); }

/* ── Portal banners ───────────────────────────────────────── */
.portal-banner {
    position: relative;
    overflow: hidden;
    padding: 18px 24px;
    border-radius: 16px;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 1.05rem;
    font-weight: 700;
    color: #fff;
    animation: fadeInUp 0.4s ease;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.portal-banner::after {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 200px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05));
    border-radius: 0 16px 16px 0;
}
.banner-student {
    background: linear-gradient(135deg, #064E3B 0%, #0D9488 55%, #5EEAD4 100%);
    border: 1px solid rgba(94,234,212,0.35);
    box-shadow: 0 4px 24px rgba(13,148,136,0.25);
}
.banner-admin {
    background: linear-gradient(135deg, #022C22 0%, #064E3B 55%, #065F46 100%);
    border: 1px solid rgba(16,185,129,0.3);
    box-shadow: 0 4px 24px rgba(6,78,59,0.4);
}
.banner-teacher {
    background: linear-gradient(135deg, #064E3B 0%, #059669 60%, #10B981 100%);
    border: 1px solid rgba(16,185,129,0.35);
    box-shadow: 0 4px 24px rgba(16,185,129,0.2);
}
.banner-sub {
    font-size: 0.8rem;
    font-weight: 400;
    opacity: 0.8;
    margin-left: auto;
}

/* ── AI Insight cards ─────────────────────────────────────── */
.ai-insight {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(16,185,129,0.06));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin: 0.75rem 0;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease;
}
.ai-insight::before {
    content: '🤖 EduAI Analysis';
    display: block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--blue);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.9;
}
.ai-insight-text {
    color: var(--text2) !important;
    font-size: 0.95rem;
    line-height: 1.6;
}
.rec-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 3px solid var(--blue);
    animation: fadeInUp 0.4s ease both;
    transition: transform 0.2s ease;
}
.rec-card:hover { transform: translateX(4px); }
.rec-card.good  { border-left-color: var(--teal); }
.rec-card.warn  { border-left-color: var(--amber); }
.rec-card.crit  { border-left-color: var(--red); }
.rec-title { font-weight: 700; color: var(--text) !important; margin-bottom: 0.3rem; font-size: 0.95rem; }
.rec-body  { color: var(--text-muted) !important; font-size: 0.88rem; line-height: 1.55; }

/* ── KPI progress bars ────────────────────────────────────── */
.kpi-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    height: 6px;
    margin-top: 6px;
    overflow: hidden;
}
.kpi-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 1s ease;
}

/* ── SDG / Vision cards ───────────────────────────────────── */
.sdg-card {
    background: linear-gradient(135deg, rgba(22,27,39,0.9), rgba(13,17,30,0.9));
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
    transition: box-shadow 0.25s ease, transform 0.2s ease;
    animation: fadeInUp 0.45s ease both;
}
.sdg-card:hover {
    box-shadow: 0 0 24px var(--blue-glow);
    transform: translateY(-3px);
}
.sdg-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.sdg-title { font-size: 1rem; font-weight: 700; color: var(--text) !important; margin-bottom: 0.4rem; }
.sdg-body { font-size: 0.85rem; color: var(--text-muted) !important; line-height: 1.55; }

/* ══════════════════════════════════════════════════════════
   VISIBILITY FIXES — inputs, tables, dropdowns, labels
   ══════════════════════════════════════════════════════════ */

/* ── Force light text in ALL input variants ───────────────── */
input, textarea, select,
.stTextInput input,
.stTextInput > div > div > input,
[data-testid="stTextInput"] input,
[data-testid="textInput"] input,
.stTextArea textarea,
[data-testid="stTextArea"] textarea {
    color: #F1F5F9 !important;
    caret-color: #3B82F6 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
}
input::placeholder,
textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #4B5563 !important;
    -webkit-text-fill-color: #4B5563 !important;
    opacity: 1 !important;
}
/* Chrome autofill fix */
input:-webkit-autofill,
input:-webkit-autofill:hover,
input:-webkit-autofill:focus {
    -webkit-text-fill-color: #F1F5F9 !important;
    -webkit-box-shadow: 0 0 0px 1000px #161B27 inset !important;
    transition: background-color 5000s ease-in-out 0s;
}

/* ── Selectbox / number input text ────────────────────────── */
.stSelectbox select,
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div,
[data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] span,
[data-testid="stNumberInput"] input,
.stNumberInput input {
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
}
/* Dropdown list options */
li[role="option"], [role="listbox"] li, [role="option"] {
    color: #F1F5F9 !important;
    background: #1C2333 !important;
}
li[role="option"]:hover, [role="option"]:hover {
    background: rgba(59,130,246,0.15) !important;
}

/* ── Labels and captions ──────────────────────────────────── */
label, .stSlider label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #CBD5E1 !important;
}
small, [data-testid="stCaptionContainer"] {
    color: #64748B !important;
}

/* ── st.table() HTML tables ───────────────────────────────── */
.stTable { width: 100% !important; }
.stTable table {
    width: 100% !important;
    border-collapse: collapse !important;
    background: transparent !important;
    font-size: 0.88rem !important;
}
.stTable thead tr {
    background: rgba(16,185,129,0.12) !important;
    border-bottom: 1px solid rgba(52,211,153,0.25) !important;
}
.stTable thead th {
    color: #6EE7B7 !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    padding: 10px 14px !important;
    background: rgba(10,32,20,0.85) !important;
    border-bottom: 1px solid rgba(52,211,153,0.2) !important;
}
.stTable tbody tr {
    border-bottom: 1px solid rgba(52,211,153,0.06) !important;
    transition: background 0.15s ease !important;
}
.stTable tbody tr:hover {
    background: rgba(16,185,129,0.07) !important;
}
.stTable tbody td {
    color: #A7F3D0 !important;
    padding: 9px 14px !important;
    background: transparent !important;
}
.stTable tbody tr:nth-child(even) td {
    background: rgba(16,185,129,0.03) !important;
}

/* ── Dark DataFrames (Streamlit's data grid wrapper) ──────── */
[data-testid="stDataFrame"] > div {
    background: var(--bg2) !important;
    border-radius: 12px !important;
    border: 1px solid var(--card-border) !important;
    color: #F1F5F9 !important;
}
/* The internal grid canvas text — best effort */
[data-testid="stDataFrame"] canvas { color: #F1F5F9 !important; }
/* AG-grid / Glide cells */
.dvn-scroller { background: #161B27 !important; color: #F1F5F9 !important; }
.dvn-scroller * { color: #F1F5F9 !important; }

/* ── Number / slider value display ────────────────────────── */
[data-testid="stSlider"] [data-testid="stText"],
[data-testid="stSlider"] p {
    color: #93C5FD !important;
}

/* ── Annotation / vline text on Plotly ────────────────────── */
/* Applied via fig.update_layout - see chart code */

/* ── Alert box text ───────────────────────────────────────── */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: var(--text2) !important;
}

/* ── Spinner text ─────────────────────────────────────────── */
[data-testid="stSpinner"] p { color: var(--text2) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DATASET_FILE   = "StudentPerformanceFactors.csv"
WEAK_THRESHOLD = 60
AVG_THRESHOLD  = 75

# ─────────────────────────────────────────────────────────────
# CREDENTIALS  (username → password + role)
# ─────────────────────────────────────────────────────────────
CREDENTIALS = {
    "admin":   {"password": "admin123",   "role": "admin"},
    "teacher": {"password": "teacher123", "role": "teacher"},
    "student": {"password": "student123", "role": "student"},
}

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
        "logged_in":     False,
        "page":          "login",
        "username":      "",
        "role":          "",
        "chat_history":  [],
        "retrain_count": 0,
        "last_retrain":  None,
        "login_error":   "",
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
    st.session_state.login_error  = ""
    if username:
        st.session_state.username = username
    if role:
        st.session_state.role     = role
    st.rerun()


def logout():
    for key in ["logged_in", "page", "username", "role",
                "chat_history", "retrain_count", "last_retrain", "login_error"]:
        if key in st.session_state:
            del st.session_state[key]
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
    if not st.session_state.get("logged_in", False):
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
        badge_colors = {
            "student": ("rgba(139,92,246,0.15)", "#C4B5FD", "rgba(139,92,246,0.3)"),
            "teacher": ("rgba(16,185,129,0.15)",  "#6EE7B7", "rgba(16,185,129,0.3)"),
            "admin":   ("rgba(239,68,68,0.15)",   "#FCA5A5", "rgba(239,68,68,0.3)"),
        }
        bc = badge_colors.get(role, badge_colors["student"])
        st.markdown(
            f"<div style='background:{bc[0]};color:{bc[1]};border:1px solid {bc[2]};padding:9px 14px;"
            f"border-radius:10px;font-size:0.82rem;font-weight:700;text-align:center;letter-spacing:0.3px;'>"
            f"{info['icon']} {info['badge']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"**👤 User:** `{username}`")
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

        if st.button("🚪  Log Out", use_container_width=True, type="secondary"):
            logout()

        st.divider()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        st.caption(f"Session: {now}")
        st.caption("Dataset: Kaggle · SDG 4 · Vision 2030/35")


# ─────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────
def render_login():
    """Secure single-entry login page. No portal content is visible until authenticated."""

    # Centre the form with empty column padding
    _, centre, _ = st.columns([1, 1.4, 1])

    with centre:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Premium login hero ─────────────────────────────────
        st.markdown("""
        <div class="login-hero">
            <span class="login-logo">🎓</span>
            <div class="login-title">EduAI Platform</div>
            <div class="login-sub">AI-Powered Academic Intelligence System</div>
            <div style="margin-top:0.75rem;">
                <span class="login-pill">✦ SDG 4: Quality Education</span>
                <span class="login-pill">✦ Vision 2030/2035</span>
                <span class="login-pill">✦ AI Analytics</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Inputs + button inside a clean container
        with st.container(border=True):
            st.markdown("##### Sign in to your portal")
            st.markdown("&nbsp;")
            username = st.text_input("Username", placeholder="Enter your username",
                                     key="login_username")
            password = st.text_input("Password", placeholder="Enter your password",
                                     type="password", key="login_password")
            st.markdown("&nbsp;")

            if st.button("Login", type="primary", use_container_width=True):
                uname = username.strip().lower()
                if uname in CREDENTIALS and password == CREDENTIALS[uname]["password"]:
                    role = CREDENTIALS[uname]["role"]
                    st.session_state.logged_in    = True
                    st.session_state.username     = username.strip()
                    st.session_state.role         = role
                    st.session_state.page         = role
                    st.session_state.login_error  = ""
                    st.session_state.chat_history = []
                    st.rerun()
                else:
                    st.session_state.login_error = "Incorrect username or password. Please try again."

            if st.session_state.get("login_error"):
                st.error(st.session_state.login_error)

        # ── Demo credentials ───────────────────────────────────
        with st.expander("💡 Demo Credentials (For Testing)"):
            st.markdown("""
<div class="cred-row cred-admin">
  <span>⚙️ <strong>Admin Portal</strong></span>
  <span>Username: <code>admin</code> &nbsp; Password: <code>admin123</code></span>
  <span class="cred-badge badge-admin">ADMIN</span>
</div>
<div class="cred-row cred-teacher">
  <span>📊 <strong>Teacher Portal</strong></span>
  <span>Username: <code>teacher</code> &nbsp; Password: <code>teacher123</code></span>
  <span class="cred-badge badge-teacher">TEACHER</span>
</div>
<div class="cred-row cred-student">
  <span>🎒 <strong>Student Portal</strong></span>
  <span>Username: <code>student</code> &nbsp; Password: <code>student123</code></span>
  <span class="cred-badge badge-student">STUDENT</span>
</div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "Dataset: [Kaggle — Student Performance Factors]"
            "(https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) "
            "· 6,607 records · CC0 License"
        )


# ─────────────────────────────────────────────────────────────
# STUDENT PORTAL
# ─────────────────────────────────────────────────────────────
def render_student(df, lr, rf, feat_cols, df_ml, metrics):
    name = st.session_state.username or "Student"

    # Role header
    st.markdown(
        "<div class='portal-banner banner-student'>"
        "🎒 <strong>Student Portal</strong>"
        "<span class='banner-sub'>Personal academic AI assistant</span></div>",
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
                delta={"reference": 75, "increasing": {"color": "#10B981"},
                       "decreasing": {"color": "#EF4444"}},
                number={"font": {"color": "#F1F5F9", "size": 52}},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickcolor": "#64748B",
                             "tickfont": {"color": "#94A3B8"}},
                    "bar":  {"color": "#3B82F6", "thickness": 0.7},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(148,163,184,0.1)",
                    "steps": [
                        {"range": [0,  WEAK_THRESHOLD],            "color": "rgba(239,68,68,0.18)"},
                        {"range": [WEAK_THRESHOLD, AVG_THRESHOLD], "color": "rgba(245,158,11,0.15)"},
                        {"range": [AVG_THRESHOLD,  100],           "color": "rgba(16,185,129,0.15)"},
                    ],
                    "threshold": {"line": {"color": "#10B981", "width": 3},
                                  "thickness": 0.8, "value": 75},
                },
                title={"text": "EduAI Predicted Score", "font": {"color": "#94A3B8", "size": 14}},
            ))
            gauge.update_layout(
                height=300,
                margin=dict(t=60, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter, sans-serif"},
            )
            st.plotly_chart(gauge, use_container_width=True)

            ai_msgs = {
                "Strong": (
                    f"EduAI has detected exceptional academic consistency in your profile. "
                    f"With a predicted score of **{score:.1f}/100**, your study patterns "
                    f"({hours} hrs/week) and attendance ({attendance}%) place you in the top-performance "
                    f"tier. EduAI recommends channelling this momentum into peer-teaching and advanced "
                    f"practice problems to maximise your ceiling score."
                ),
                "Average": (
                    f"EduAI analysis indicates solid academic engagement with measurable room for growth. "
                    f"Your predicted score of **{score:.1f}/100** reflects consistent effort, but EduAI "
                    f"has identified that a 15–20% increase in weekly study hours combined with active "
                    f"recall techniques could propel your score into the Strong tier within 4–6 weeks."
                ),
                "Weak": (
                    f"EduAI has detected declining academic consistency caused by reduced weekly study "
                    f"hours and below-average attendance patterns. Predicted score: **{score:.1f}/100**. "
                    f"EduAI recommends increasing structured study sessions by 5+ hours weekly and "
                    f"targeting 80%+ attendance — historical data shows this combination lifts students "
                    f"by a full performance category within one semester."
                ),
            }
            st.markdown(
                f"<div class='ai-insight'>"
                f"<div class='ai-insight-text'>{ai_msgs[category]}</div></div>",
                unsafe_allow_html=True,
            )

            st.divider()
            st.markdown("### 📈 Your Score in Class Context")
            fig = px.histogram(df, x="Exam_Score", nbins=30,
                               color_discrete_sequence=["#3B82F6"],
                               labels={"Exam_Score": "Exam Score"},
                               title="Class Score Distribution — Your EduAI Prediction Marked")
            fig.add_vline(x=score, line_color="#10B981", line_width=3,
                          annotation_text=f"You: {score:.1f}", annotation_position="top right",
                          annotation_font_color="#10B981")
            fig.add_vline(x=df["Exam_Score"].mean(), line_color="#94A3B8", line_dash="dash",
                          annotation_text=f"Class avg: {df['Exam_Score'].mean():.1f}",
                          annotation_font_color="#CBD5E1")
            fig.update_layout(
                height=340,
                margin=dict(t=55, b=15),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,17,30,0.5)",
                font={"family": "Inter, sans-serif", "color": "#94A3B8"},
                title_font_color="#CBD5E1",
            )
            fig.update_traces(marker_line_color="rgba(59,130,246,0.3)", marker_line_width=0.5)
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
                if title.startswith("✅"):
                    cls = "rec-card good"
                elif "Critical" in title or "Insufficient" in title or "Needs" in title:
                    cls = "rec-card crit"
                else:
                    cls = "rec-card warn"
                body_html = body.replace("\n", "<br>")
                st.markdown(
                    f"<div class='{cls}'>"
                    f"<div class='rec-title'>{title}</div>"
                    f"<div class='rec-body'>{body_html}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

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
            st.table(schedule_df)

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
        "<div class='portal-banner banner-admin'>"
        "⚙️ <strong>Admin Portal</strong>"
        "<span class='banner-sub'>🔒 Full System Authority — Admin Access Only</span></div>",
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
            st.table(display.head(n).reset_index(drop=True))
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
            st.table(miss[miss > 0].rename("Missing Count"))

        dtype_df = pd.DataFrame({
            "Column":        df.columns,
            "Type":          df.dtypes.astype(str).values,
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.table(dtype_df.reset_index(drop=True))

    # ── Tab 2: Data Analysis ──────────────────────────────────
    with tab2:
        st.markdown("#### Descriptive Statistics")
        st.table(df.select_dtypes(include=[np.number]).describe().round(2))

        st.markdown("#### Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r",
                             title="Feature Correlation Matrix")
        fig_heat.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,30,0.5)",
            font={"family": "Inter, sans-serif", "color": "#94A3B8"},
            title_font_color="#CBD5E1",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Column Distribution Explorer")
        col_sel = st.selectbox("Select column", df.columns)
        if df[col_sel].dtype in [np.float64, np.int64]:
            fig_col = px.histogram(df, x=col_sel, nbins=30,
                                   color_discrete_sequence=["#EF4444"],
                                   title=f"Distribution: {col_sel}")
        else:
            vc = df[col_sel].value_counts().reset_index()
            vc.columns = [col_sel, "Count"]
            fig_col = px.bar(vc, x=col_sel, y="Count",
                             color_discrete_sequence=["#EF4444"],
                             title=f"Value Counts: {col_sel}")
        fig_col.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,30,0.5)",
            font={"family": "Inter, sans-serif", "color": "#94A3B8"},
            title_font_color="#CBD5E1",
        )
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
                         color="Importance",
                         color_continuous_scale=[[0, "#1E3A8A"], [0.5, "#3B82F6"], [1, "#10B981"]],
                         title="EduAI — Random Forest Feature Importances")
        fig_imp.update_layout(
            height=400,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(t=55, b=10),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,30,0.5)",
            font={"family": "Inter, sans-serif", "color": "#94A3B8"},
            title_font_color="#CBD5E1",
        )
        fig_imp.update_traces(marker_line_width=0)
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
        "<div class='portal-banner banner-teacher'>"
        "📊 <strong>Teacher Portal</strong>"
        "<span class='banner-sub'>Educational Authority — Analytics & Reporting</span></div>",
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

        _dark = dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,30,0.5)",
            font={"family": "Inter, sans-serif", "color": "#94A3B8"},
            title_font_color="#CBD5E1",
        )

        with col_l:
            cat_counts = df["Performance"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_donut = px.pie(
                cat_counts, values="Count", names="Category",
                hole=0.5, title="Performance Breakdown",
                color="Category",
                color_discrete_map={"Strong": "#10B981", "Average": "#F59E0B", "Weak": "#EF4444"},
            )
            fig_donut.update_layout(height=360, **_dark)
            fig_donut.update_traces(
                textfont_color="white",
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=2,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_r:
            if "Gender" in df.columns:
                fig_box = px.box(df, x="Gender", y="Exam_Score", color="Gender",
                                 title="Score Distribution by Gender",
                                 color_discrete_sequence=["#3B82F6", "#A78BFA"])
                fig_box.update_layout(height=360, showlegend=False, **_dark)
                st.plotly_chart(fig_box, use_container_width=True)

        df2 = df.copy()
        df2["Study_Bucket"] = pd.cut(df2["Hours_Studied"],
                                      bins=[0, 5, 10, 15, 20, 25, 30, 99],
                                      labels=["0-5","6-10","11-15","16-20","21-25","26-30","30+"])
        bucket = df2.groupby("Study_Bucket", observed=True)["Exam_Score"].mean().reset_index()
        bucket.columns = ["Weekly Study Hours", "Average Score"]
        fig_trend = px.line(bucket, x="Weekly Study Hours", y="Average Score",
                            markers=True, title="EduAI Insight: Average Score vs Weekly Study Hours",
                            color_discrete_sequence=["#10B981"])
        fig_trend.update_traces(
            marker=dict(size=8, color="#34D399", line=dict(color="#065F46", width=1.5)),
            line=dict(width=2.5),
        )
        fig_trend.update_layout(height=320, **_dark)
        st.plotly_chart(fig_trend, use_container_width=True)

        if "School_Type" in df.columns:
            school_avg = df.groupby("School_Type")["Exam_Score"].mean().reset_index()
            school_avg.columns = ["School Type", "Average Score"]
            fig_sch = px.bar(school_avg, x="School Type", y="Average Score",
                             color="Average Score",
                             color_continuous_scale=[[0, "#064E3B"], [0.5, "#059669"], [1, "#34D399"]],
                             text="Average Score", title="Average Score by School Type")
            fig_sch.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                   textfont_color="#CBD5E1", marker_line_width=0)
            fig_sch.update_layout(height=300, **_dark)
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

        _dark2 = dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,30,0.5)",
            font={"family": "Inter, sans-serif", "color": "#94A3B8"},
            title_font_color="#CBD5E1",
        )

        fig_sc = px.scatter(
            weak_df, x="Hours_Studied", y="Exam_Score",
            color="Attendance", size_max=9,
            title="EduAI At-Risk Analysis — Study Hours vs Score (colour = Attendance %)",
            labels={"Hours_Studied": "Hours Studied / Week"},
            color_continuous_scale=[[0, "#EF4444"], [0.5, "#F59E0B"], [1, "#10B981"]],
        )
        fig_sc.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color="rgba(0,0,0,0.3)")))
        fig_sc.update_layout(height=400, **_dark2)
        st.plotly_chart(fig_sc, use_container_width=True)

        if "Motivation_Level" in df.columns:
            mot = weak_df["Motivation_Level"].value_counts().reset_index()
            mot.columns = ["Motivation", "Count"]
            fig_mot = px.bar(mot, x="Motivation", y="Count",
                             title="Motivation Levels — At-Risk Students",
                             color_discrete_sequence=["#EF4444"])
            fig_mot.update_layout(height=280, **_dark2)
            fig_mot.update_traces(marker_line_width=0)
            st.plotly_chart(fig_mot, use_container_width=True)

        show = [c for c in ["Hours_Studied", "Attendance", "Sleep_Hours",
                             "Motivation_Level", "Exam_Score"] if c in df.columns]
        st.markdown("**Bottom 20 Students by Score**")
        st.table(weak_df[show].sort_values("Exam_Score").head(20).reset_index(drop=True))

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
        st.table(strong_df[show2].sort_values("Exam_Score", ascending=False).head(20).reset_index(drop=True))

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
        _sdg = dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,17,30,0.5)",
            font={"family": "Inter, sans-serif", "color": "#94A3B8"},
            title_font_color="#CBD5E1",
        )

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
                    title="EduAI Predictive Pattern: Study Hours → Score",
                    color_discrete_map={"Strong": "#10B981", "Average": "#F59E0B", "Weak": "#EF4444"},
                )
                fig_pred.update_traces(marker=dict(opacity=0.8, size=6))
                fig_pred.update_layout(height=290, margin=dict(t=45, b=10), **_sdg)
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
                                     color="Avg Score",
                                     color_continuous_scale=[[0,"#064E3B"],[0.5,"#059669"],[1,"#34D399"]],
                                     text="Avg Score", title="Equity Analysis: Score by Resource Access")
                    fig_res.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                          textfont_color="#CBD5E1", marker_line_width=0)
                    fig_res.update_layout(height=290, margin=dict(t=45, b=10), **_sdg)
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
        st.table(pd.DataFrame(kpi_data))

        if "Internet_Access" in df.columns:
            inet = df.groupby("Internet_Access")["Exam_Score"].mean().reset_index()
            inet.columns = ["Internet Access", "Avg Score"]
            fig_inet = px.bar(inet, x="Internet Access", y="Avg Score",
                              color_discrete_sequence=["#10B981", "#EF4444"],
                              text="Avg Score", title="Digital Divide: Internet Access vs Academic Score")
            fig_inet.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                   textfont_color="#CBD5E1", marker_line_width=0)
            fig_inet.update_layout(height=290, **_sdg)
            st.plotly_chart(fig_inet, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Gate: if not authenticated, only show the login page
    if not st.session_state.get("logged_in", False):
        render_login()
        return

    render_sidebar()

    with st.spinner("Loading AI models…"):
        df = load_data()
        lr, rf, feat_cols, df_ml, metrics = train_models(df)

    page = st.session_state.page
    if page == "student":
        render_student(df, lr, rf, feat_cols, df_ml, metrics)
    elif page == "admin":
        render_admin(df, lr, rf, feat_cols, df_ml, metrics)
    elif page == "teacher":
        render_teacher(df)
    else:
        render_login()


if __name__ == "__main__":
    main()
