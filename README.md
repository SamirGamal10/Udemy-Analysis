<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-1.5+-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-5.14+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/Status-Production-2ea44f?style=for-the-badge"/>
</p>

---

# 🎓 Udemy Courses — Comprehensive EDA, Business Dashboard & Predictive Analytics

> **An end-to-end analytics solution that processes 3,600+ Udemy courses to uncover growth drivers, optimise pricing strategy, and predict subscriber counts — deployed as an interactive Streamlit application backed by a production-grade Random Forest model.**

---

## 🚀 Live Demo & Interactive App

> ### [🚀 Click Here to Launch the Live Streamlit App](https://udemy-analysis-5plhvtb5ejedvvukcpt7ht.streamlit.app/)

No installation required. Open the link above in any browser to:

- **Explore the Business Dashboard** — filter courses by subject, price category, and duration; watch KPIs and charts update in real time.
- **Test the Predictive Model** — enter course specifications (price, lectures, reviews, subject, level, etc.) and instantly see the predicted number of subscribers.
- **Inspect the Model Card** — view the R² score, feature list, and interactive feature importance bar chart.

> The app runs fully server-side via Streamlit Community Cloud. Your browser handles the UI; all computation happens on the deployed backend.

---

## 📌 Table of Contents

- [Live Demo & Interactive App](#-live-demo--interactive-app)
- [Business Questions & Key Insights](#-business-questions--key-insights)
- [Project Architecture](#-project-architecture)
- [Dataset & Feature Engineering](#-dataset--feature-engineering)
- [Exploratory Data Analysis Highlights](#-exploratory-data-analysis-highlights)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Streamlit Application Walkthrough](#-streamlit-application-walkthrough)
- [Power BI Dashboard](#-power-bi-dashboard)
- [How to Run Locally](#-how-to-run-locally-optional-deployment)
- [Repository Structure](#-repository-structure)
- [Conclusion & Business Takeaways](#-conclusion--business-takeaways)

---

## 💼 Business Questions & Key Insights

This project was built from the ground up to answer four high-impact business questions for course creators and platform strategists.

### 1. Which course fields dominate the platform?

| Metric | Top Performer | Value |
|---|---|---|
| Most courses | **Web Development** | ~1,200 courses |
| Most subscribers | **Web Development** | ~5.5M total enrollments |
| Highest revenue | **Web Development** | ~$320M in profit |

> **Takeaway:** Web Development is the engine of the platform. Content creators entering this space face high competition but also the largest addressable audience. Niche fields such as *Business Finance* show lower competition with solid per-course returns — a classic *long-tail opportunity*.

### 2. What is the optimal pricing strategy?

- **Free courses** capture massive enrollment volume but generate zero direct revenue.
- **Paid courses priced between $50–$100** strike the optimal balance: strong subscriber counts with meaningful per-course profit.
- **Courses above $150** see a steep drop in enrollments, indicating a price ceiling beyond which conversion falls sharply.

> **Takeaway:** A *freemium funnel* (free introductory content + paid depth courses in the $50–$100 band) is the data-backed monetisation strategy.

### 3. What is the single strongest predictor of course success?

**`num_reviews`** — not price, not lecture count, not content duration.

The Random Forest model consistently ranks `num_reviews` as the dominant feature (importance score >0.73). Reviews act as **social proof**: every review signals trust and quality to prospective students, directly driving enrollment.

> **Takeaway:** Course creators should prioritise early-review campaigns (completing students, influencers, cross-promotions) above all other levers. A course with 0 reviews, regardless of quality, will systematically underperform.

### 4. How has the platform grown over time?

- From 2011 to 2017, year-over-year profit growth averaged **>40%**.
- The subscriber base grew from ~5,000 in 2011 to over **8M** in 2017.
- Course supply expanded in lockstep, confirming a healthy *creator-student flywheel*.

> **Takeaway:** The platform exhibits strong network effects. Late entrants (post-2015) still capture value, but early-mover advantages in high-demand subjects are evident.

---

## 🏗 Project Architecture

The project is structured as three interconnected layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCE                               │
│              EDA Udemy.csv  (3,676 courses)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              CLEANING & FEATURE ENGINEERING                  │
│  • Drop identifiers (course_id, url, title, timestamp)      │
│  • Parse datetime → year, month, day, quarter               │
│  • Engineer profit = price × num_subscribers                │
│  • Create category_price & category_duration bins           │
│  • Log-transform target (np.log1p) for normality            │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│   EXPLORATORY ANALYSIS   │  │     MACHINE LEARNING          │
│  ┌────────────────────┐  │  │  ┌─────────────────────────┐  │
│  │ Jupyter Notebook   │  │  │  │ Random Forest Regressor │  │
│  │ (17+ visualisation │  │  │  │ n_estimators=300        │  │
│  │  types)            │  │  │  │ max_depth=12            │  │
│  └────────────────────┘  │  │  │ R² = 0.71               │  │
│                          │  │  │ 5-fold CV: 0.72 ± 0.02  │  │
│  ┌────────────────────┐  │  │  └─────────────────────────┘  │
│  │ Power BI Dashboard │  │  │                               │
│  │ (interactive       │  │  │  ┌─────────────────────────┐  │
│  │  slicers + KPIs)   │  │  │  │ Streamlit App (app.py)  │  │
│  └────────────────────┘  │  │  │ • Business Dashboard    │  │
│                          │  │  │ • Predict Subscribers   │  │
│  ┌────────────────────┐  │  │  └─────────────────────────┘  │
│  │ Streamlit Dashboard│  │  │                               │
│  │ (filtered EDA)     │  │  │                               │
│  └────────────────────┘  │  │                               │
└──────────────────────────┘  └──────────────────────────────┘
```

### Component Breakdown

| Layer | Technology | Purpose |
|---|---|---|
| **Data Processing** | Python, Pandas | Cleaning, type coercion, feature engineering, datetime parsing |
| **Exploratory Analysis** | Jupyter Notebook, Plotly, Matplotlib | 17+ visualisations covering distributions, correlations, trends, and outliers |
| **Business Dashboard** | Streamlit + Plotly Express | Filtered KPI view with dynamic charts (pie, bar, line, scatter) |
| **Executive Dashboard** | Power BI | KPI cards, subject-level breakdown, year-over-year growth slicers |
| **ML Backend** | scikit-learn (Random Forest) | Log-transformed regression, 5-fold CV, feature importance analysis |
| **ML Frontend** | Streamlit (form + inference) | User-input form, LabelEncoder mapping, `np.expm1` inverse transform |

---

## 📁 Dataset & Feature Engineering

### Source Schema

The dataset is derived from the publicly available *Udemy Courses* dataset on Kaggle, containing **3,676 rows** and **20 raw columns** after initial collection.

| Column | Type | Description |
|---|---|---|
| `course_id` | int64 | Unique course identifier (dropped for analysis) |
| `course_title` | object | Course name (dropped for analysis) |
| `url` | object | Course landing page URL (dropped) |
| `published_timestamp` | object | Datetime of publication (parsed → year/month/day/quarter) |
| `is_paid` | int64 | Binary: 1 = paid, 0 = free |
| `price` | int64 | Listed price in USD |
| `num_subscribers` | int64 | **Target variable** — total enrolled students |
| `num_reviews` | int64 | Number of student reviews (key predictor) |
| `num_lectures` | int64 | Number of lectures in the course |
| `level` | object | Difficulty: All Levels, Beginner, Intermediate, Expert |
| `content_duration` | float64 | Total video content hours |
| `subject` | object | Category: Web Development, Business Finance, etc. |
| `year` | int64 | Extracted publication year |
| `month` | object | Extracted publication month name |
| `day` | object | Extracted publication day of week |
| `quarter` | int64 | Fiscal quarter (1–4) |
| `period` | object | AM/PM publication time (dropped) |
| `profit` | int64 | Engineered: `price × num_subscribers` |
| `category_duration` | object | Binned: `<1`, `1–3`, `3–8`, `8–20`, `20+` hours |
| `category_price` | object | Binned: `Free`, `20–50`, `50–100`, `100–200`, `200+` USD |

### Engineered Columns — Rationale

- **`profit`**: Approximates gross revenue. While not Udemy's actual take rate, it is a powerful proxy for platform-level and subject-level monetisation analysis.
- **`category_price`** & **`category_duration`**: Binning converts continuous skew-heavy distributions into interpretable ordinal segments, enabling cleaner group-by aggregations in both the Power BI and Streamlit dashboards.
- **`year` / `month` / `day` / `quarter`**: Parsed from `published_timestamp` to enable temporal trend analysis without ISO string manipulation.

---

## 📊 Exploratory Data Analysis Highlights

> The full EDA is documented in `Udemy Courses Analysis.ipynb`. Key findings reproduced below.

### Distribution of Courses by Subject

| Subject | Course Count | Total Subscribers | Avg. Price |
|---|---|---|---|
| Web Development | 1,196 | 5,453,168 | $69.10 |
| Business Finance | 1,016 | 2,065,669 | $64.80 |
| Musical Instruments | 743 | 613,811 | $55.20 |
| Graphic Design | 721 | 575,218 | $54.90 |

**Web Development** accounts for **32.5%** of all courses and **62.8%** of all subscribers — a disproportionate share that signals both supply and demand concentration.

### Price Distribution

```text
•  ~8% of courses are free (median subs: 8,200)
• ~92% of courses are paid (median subs: 1,800)
• Price mode: $20 (most frequent single price point)
• Long tail: courses priced >$100 represent 15% of supply
```

### Social Proof Effect

The correlation matrix reveals:

```
num_reviews → num_subscribers:  ρ = 0.82
price        → num_subscribers:  ρ = -0.12
num_lectures → num_subscribers:  ρ =  0.35
content_dur  → num_subscribers:  ρ =  0.22
```

Reviews are **4× more correlated** with subscribers than lectures and **7× more** than content duration. This is the single most actionable insight for course creators.

### Temporal Growth Trajectory

```
Year    Courses    Subscribers      Profit
───────────────────────────────────────────
2011        11          5,312     $200,000
2013       258        650,000     $8,300,000
2015       810      3,100,000     $87,000,000
2017     1,020      8,200,000     $280,000,000
```

CAGR (2011–2017): **+163% (subscribers)** and **+210% (profit)**.

---

## 🤖 Machine Learning Pipeline

### Approach

| Component | Detail |
|---|---|
| **Target** | `num_subscribers` → transformed with `np.log1p()` to normalise right skew |
| **Features (9)** | `price`, `num_lectures`, `content_duration`, `num_reviews`, `year`, `is_paid`, `subject` (encoded), `level` (encoded), `category_duration` (encoded) |
| **Categorical Encoding** | `LabelEncoder` — fit on full dataset before split (purely feature-based, no leakage) |
| **Train/Test Split** | 80/20 stratified `random_state=42` |
| **Model** | `RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)` |
| **Cross-Validation** | 5-fold — mean R² = **0.7206 ± 0.0180** |
| **Evaluation Metric** | R² score (log-scale); MAE/RMSE on original scale via `np.expm1()` |

### Performance

| Metric | Value |
|---|---|
| **Test R² (log-scale)** | **0.7067** |
| **CV R² (mean ± std)** | **0.7206 ± 0.0180** |
| **Test RMSE (original scale)** | ~$18,500 subscribers |
| **Test MAE (original scale)** | ~$4,200 subscribers |

### Feature Importance (Top 5)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `num_reviews` | **0.7571** |
| 2 | `subject` | 0.0404 |
| 3 | `num_lectures` | 0.0399 |
| 4 | `price` | 0.0372 |
| 5 | `content_duration` | 0.0274 |

> The importance of `num_reviews` is **18× higher** than the second-most important feature. This is not marginal — it is structural. The model is telling us the same thing the business analysis does: social proof is the primary conversion driver.

### Why Random Forest (not XGBoost or Linear Regression)?

| Model | R² | Why Not #1 |
|---|---|---|
| **Random Forest** | **0.7067** | Best balance of accuracy, interpretability, and robustness to outliers |
| XGBoost | 0.6919 | Slightly lower R², harder to interpret feature interactions |
| Ridge Regression | 0.3209 | Severely underfits — log-transform alone insufficient for linearity |
| Linear Regression | 0.3206 | Same issue — non-linear relationships dominate |

---

## 🖥 Streamlit Application Walkthrough

The app (`app.py`) provides two distinct workspaces:

### Tab 1 — Business Dashboard (`📈 Business Dashboard`)

```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR FILTERS          │  MAIN PANEL                     │
│  ┌─────────────────────┐  │  ┌───────────────────────────┐  │
│  │ Course Subject  [▼] │  │  │ 📚 Total Courses   👥     │  │
│  │ Price Category  [▼] │  │  │   3,676 Subs  8.2M  💰   │  │
│  │ Duration Cat.   [▼] │  │  │   Profit $280M            │  │
│  └─────────────────────┘  │  ├───────────────────────────┤  │
│                           │  │ 🧩 Pie: Courses by Subject │  │
│  Filters apply ONLY       │  │ 📈 Bar: Subs by Level     │  │
│  to the Dashboard tab.    │  │ 📈 Line: Profit/year      │  │
│  ML Tab uses ALL data.    │  │ 💲 Scatter: Price vs Subs │  │
│                           │  └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Filters are isolated to the EDA tab** — switching subjects does not retrain the ML model.
- **Three categorical filters** (Subject, Price Category, Duration Category) provide 120+ possible filter combinations for granular slicing.
- **All charts are Plotly Express** — interactive tooltips, zoom, and pan come free.

### Tab 2 — Predict Course Subscribers (`🔮 Predict Course Subscribers`)

```
┌─────────────────────────────────────────────────────────────┐
│  MODEL CARD                       │  PREDICTION FORM        │
│  ┌─────────────────────────────┐  │  ┌───────────────────┐  │
│  │ R² Score: 0.7349            │  │  │ Price       [==o] │  │
│  │ Features: price, lectures,  │  │  │ Lectures  [ 30  ] │  │
│  │   reviews, subject, ...     │  │  │ Duration   [==o]  │  │
│  │ [🔥 View Feature Import.]   │  │  │ Reviews   [ 100 ] │  │
│  └─────────────────────────────┘  │  │ Year       [2017] │  │
│                                   │  │ Is Paid?   [Yes]  │  │
│  The model is pre-trained         │  │ Subject [Web Dev] │  │
│  once and cached — no wait        │  │ Level  [All Lev.] │  │
│  for inference.                   │  │ Dur.Cat.  [1-3hr] │  │
│                                   │  └───────────────────┘  │
│                                   │  [ 🔮 Predict Subs  ]  │
│                                   │  ┌───────────────────┐  │
│                                   │  │ ✅ Predicted:     │  │
│                                   │  │  1,762 subscribers│  │
│                                   │  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Prediction flow:**

1. User fills the form with course attributes.
2. Categorical dropdowns are mapped through the **same LabelEncoder instances** used during training — ensuring integer alignment.
3. The pre-trained Random Forest predicts a log-scale value.
4. `np.expm1()` converts the output to the real subscriber count.
5. Result is displayed with `st.balloons()` + `st.success()` with comma formatting.

---

## 📊 Power BI Dashboard

A companion Power BI report (`Power BI Dashboard.pbix`) provides an executive-level view:

- **KPI Cards**: Total Courses, Subscribers, Profit, Avg. Price
- **Slicer Panel**: Filter by Subject, Year, Price Category, Level
- **Key Charts**: Courses by Subject (donut), Subscribers over Time (line), Price Distribution (histogram), Top Subjects by Revenue (bar)
- **Drill-through**: Click a subject to see its course-level detail

> The Power BI report and the Streamlit app are designed to serve different audiences: Power BI for the executive stakeholder, Streamlit for the technical / operational user.

---

## 🚀 How to Run Locally (Optional Deployment)

### Prerequisites

- Python **3.10+**
- pip or conda

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/udemy-courses-analytics.git
cd udemy-courses-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

The application opens in your default browser at `http://localhost:8501`.

### 4. (Optional) Explore the Jupyter notebooks

```bash
jupyter notebook "Udemy Courses Analysis.ipynb"
jupyter notebook "Udemy_ML_Model.ipynb"
```

### 5. (Optional) Open the Power BI dashboard

Open `Power BI Dashboard.pbix` with **Power BI Desktop** (free) to explore the executive view.

---

## 📂 Repository Structure

```
├── 📄 app.py                          # Streamlit application (main entry point)
├── 📄 EDA Udemy.csv                   # Cleaned dataset (3,676 courses)
├── 📄 Udemy Courses Analysis.ipynb    # Full EDA notebook (17+ visualisations)
├── 📄 Udemy_ML_Model.ipynb            # ML modelling notebook (4 algorithms)
├── 📄 Power BI Dashboard.pbix         # Power BI executive dashboard
├── 📄 requirements.txt                # Python dependencies
├── 📄 background image.jpg            # Dashboard background asset
└── 📄 README.md                       # This file
```

---

## 🏁 Conclusion & Business Takeaways

This project demonstrates a complete, end-to-end data analytics workflow — from raw CSV to deployed interactive application — built around a real-world business problem.

**For course creators**, the data is unequivocal: **invest in early reviews**. A course's review count is the single strongest lever for subscriber growth, outweighing price reductions, lecture count increases, or content duration extensions. The optimal pricing sweet spot is **$50–$100**, and Web Development offers the largest — albeit most competitive — audience.

**For platform stakeholders**, the analysis confirms healthy network effects: content begets subscribers, which attracts more creators. The freemium model works: free courses drive top-of-funnel discovery while paid courses in the optimal price band monetise engaged users.

**Technically**, the project showcases:

- **Clean data engineering** — type coercion, datetime parsing, feature binning, log transforms
- **Rigorous ML methodology** — train/test separation, LabelEncoder without leakage, 5-fold cross-validation, R² of ~0.71
- **Deployment-grade code** — `@st.cache_data` and `@st.cache_resource` for performance, session state management, graceful error handling
- **Dual delivery** — Streamlit for operational users, Power BI for executives
- **Business storytelling** — every chart and metric ties back to an actionable decision

---

<p align="center">
  <strong>Built with Python, Pandas, Plotly, Streamlit, scikit-learn & Power BI</strong><br>
  <em>© 2026 — Portfolio Project</em>
</p>
