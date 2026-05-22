# 🎓 AI Student Performance Assistant

> A professional, role-based AI-powered educational management platform built with Streamlit, scikit-learn, and Plotly — supporting **SDG 4: Quality Education**, **Vision 2030**, and **Vision 2035**.

---

## 🌍 SDG 4 — Quality Education

**UN Sustainable Development Goal 4** calls to *"ensure inclusive and equitable quality education and promote lifelong learning opportunities for all."*

This platform advances SDG 4 by:

- **Identifying at-risk students** before they fall too far behind, enabling early intervention.
- **Personalising learning recommendations** using AI so every student gets targeted advice.
- **Empowering educators** with data-driven dashboards that highlight equity gaps (internet access, resource availability, parental involvement).
- **Measuring institutional progress** toward SDG 4 targets in real time through the Client Portal.

---

## 🚀 Vision 2030 / Vision 2035 Alignment

| Initiative | How This Platform Contributes |
|---|---|
| **Vision 2030** | Data-driven education reform; reducing at-risk student rates below 15%; ensuring universal access to learning resources |
| **Vision 2035** | Building a knowledge-based economy through AI-powered personalised learning; closing the digital divide in education |

The Client Portal tracks live progress against Vision 2030/2035 KPIs including at-risk rate, average study hours, and attendance benchmarks.

---

## 📂 Dataset

**Source:** [Kaggle — Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)  
**License:** CC0 Public Domain  
**File:** `StudentPerformanceFactors.csv`

| Property | Value |
|---|---|
| Total Records | 6,607 students |
| Features | 20 columns |
| Target Variable | `Exam_Score` (0–100) |

**Key features used:**

| Feature | Description |
|---|---|
| `Hours_Studied` | Weekly study hours |
| `Attendance` | Class attendance percentage |
| `Sleep_Hours` | Average nightly sleep |
| `Previous_Scores` | Prior academic performance |
| `Motivation_Level` | Low / Medium / High |
| `Parental_Involvement` | Low / Medium / High |
| `Access_to_Resources` | Low / Medium / High |
| `Internet_Access` | Yes / No |
| `Teacher_Quality` | Low / Medium / High |
| `Exam_Score` | Final exam score (target) |

---

## 🔐 Role-Based Dashboards

### 🏠 Landing Page
- Platform overview with SDG 4, Vision 2030/2035 alignment cards
- Live KPI metrics (students analysed, ML accuracy, at-risk rate)
- Portal entry cards for all three roles
- Tech stack overview and dataset attribution

### 🎒 Student Portal
Students access their personalised AI academic assistant:
- **Login simulation** — select a student profile
- **AI Performance Predictor** — input study hours, attendance, sleep, motivation → get predicted exam score and performance category
- **Gauge chart** — visual score prediction with Weak/Average/Strong zones
- **AI-generated feedback** — personalised narrative based on predicted category
- **Personalised study recommendations** — specific advice for each input factor
- **Class distribution chart** — see where you stand relative to the class

### ⚙️ Admin Portal
Technical staff manage the system:
- **System status cards** — online status, record count, missing values, active models
- **Dataset viewer** — paginated raw data with search/filter and CSV download
- **Descriptive statistics** — full `.describe()` output
- **Correlation heatmap** — interactive Plotly heatmap of all numeric features
- **Data health checks** — missing value analysis, data type overview, column distribution viewer
- **Dataset reload** — clear cache and re-read CSV from disk
- **ML model metrics** — LR RMSE, R² score, RF accuracy, feature importance bar chart
- **Kaggle dataset attribution** — source, license, and column info

### 📊 Client Portal
School administrators and teachers see:
- **KPI row** — total students, strong/average/weak counts with percentages
- **Performance breakdown donut chart**
- **Score distribution by gender** (box plot)
- **Average score by school type**
- **Score trend vs weekly study hours**
- **At-risk student monitoring** — scatter plots, motivation breakdown, sample table
- **Top performer analysis** — by school type and parental involvement
- **Factor analysis** — scatter matrix and box plots for categorical factors
- **SDG 4 Impact tab** — resource access gap, internet access gap, learning disability inclusion, Vision 2030/2035 progress tracker

---

## 🤖 AI / ML Functionality

### Models

| Model | Type | Input | Output | Metric |
|---|---|---|---|---|
| **Linear Regression** | Regression | All 19 features | Predicted exam score | RMSE ≈ 3.3, R² ≈ 0.18 |
| **Random Forest Classifier** | Classification | All 19 features | Weak / Average / Strong | Accuracy ≈ 97.2% |

### AI Features
- **Score prediction** with gauge chart visualisation
- **Category classification** with actionable feedback
- **Feature importance ranking** to identify the most influential factors
- **Personalised narrative feedback** mapped to predicted performance category
- **Mock LMSYS / LM Arena API wrapper** — commented integration point for connecting a real LLM (see `app.py` home page section)

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Frontend | Streamlit | 1.35.0 |
| Charts | Plotly | 5.22.0 |
| Data processing | Pandas | 2.2.2 |
| Numerical computing | NumPy | 1.26.4 |
| Machine learning | scikit-learn | 1.4.2 |
| Language | Python | 3.11 |

---

## 📁 Project Structure

```
student-assistant/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── StudentPerformanceFactors.csv # Dataset (6,607 records)
```

---

## ▶️ How to Run

### On Replit

1. Open the **Shell** tab.
2. Navigate to the project folder:
   ```bash
   cd student-assistant
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py --server.port 5000
   ```
5. The app opens automatically in the Replit preview pane.

> **Tip:** On Replit, the app is pre-configured to run automatically via a workflow — just click **Run**.

### Locally (macOS / Linux / Windows)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd student-assistant

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure the dataset is present
#    Download from: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
#    Place StudentPerformanceFactors.csv in the student-assistant/ folder

# 5. Run the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🌐 Deployment

### Streamlit Community Cloud
1. Push the `student-assistant/` folder contents to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set the main file path to `app.py`.
4. Deploy — it's free for public repos.

### Vercel (Static wrapper approach)
Vercel does not natively run Python. To deploy on Vercel:

1. Use `vercel-python` serverless functions or wrap the app with a Docker container.
2. Alternatively, deploy on **Railway**, **Render**, or **Fly.io** which support Python natively:

**Render.com** (recommended free option):
```bash
# In your repo root, create render.yaml:
services:
  - type: web
    name: ai-student-assistant
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.headless true
```

**Railway:**
```bash
railway up
# Set start command: streamlit run app.py --server.port $PORT --server.headless true
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

---

## 📸 Portal Screenshots

| Portal | Description |
|---|---|
| 🏠 Home | Landing page with SDG 4 cards, KPIs, and role selection |
| 🎒 Student | AI score predictor with gauge chart and personalised feedback |
| ⚙️ Admin | Dataset viewer, correlation heatmap, ML metrics |
| 📊 Client | At-risk monitoring, performance breakdown, SDG 4 progress |

---

## 🏆 Why This Project?

This platform demonstrates:
- **Real-world ML application** in education using a public Kaggle dataset
- **Role-based UX design** with distinct dashboards for three user types
- **SDG-aligned innovation** — measurable contribution to Quality Education
- **Production-ready architecture** — cached ML models, clean data pipeline, modular code
- **University presentation quality** — suitable for live demo, GitHub submission, and SDG hackathon showcase

---

*Built with Python · Streamlit · scikit-learn · Plotly · Supporting SDG 4 Quality Education · Vision 2030 / 2035 Aligned*
