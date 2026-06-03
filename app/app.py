"""
Udemy Courses — Streamlit Application
======================================
Two-tab production app:
  - Tab 1: Business Dashboard (EDA with sidebar filters)
  - Tab 2: Predict Course Subscribers (static Random Forest)
"""

import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd
import numpy as np
import plotly.express as px # pyright: ignore[reportMissingImports]

from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestRegressor # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import LabelEncoder # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import r2_score # pyright: ignore[reportMissingModuleSource]

import os
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Udemy Business Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
COLOR_SEQUENCE = px.colors.qualitative.Set2
CONTINUOUS_COLORS = "Tealgrn"
PRIMARY_COLOR = "#2ca02c"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../data/processed/EDA Udemy.csv")

DROP_COLS = ["course_id", "url", "course_title", "published_timestamp", "period"]
ML_DROP_COLS = ["profit", "category_price"]

NUM_FEATURES = ["price", "num_lectures", "content_duration", "num_reviews", "year", "is_paid"]
CAT_FEATURES = ["subject", "level", "category_duration"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES  # 9 features

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load CSV and drop identifier columns."""
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df

# ---------------------------------------------------------------------------
# CACHED MODEL TRAINING (static — no user-facing params)
# ---------------------------------------------------------------------------
@st.cache_resource
def train_static_model():
    """
    Train the Random Forest model behind the scenes.
    Returns: (model, encoders, feature_order, r2_score, feature_importance_df)
    """
    df = load_data()
    df_ml = df.drop(columns=[c for c in ML_DROP_COLS if c in df.columns], errors="ignore")

    # LabelEncode categoricals on the full dataset
    encoders = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le

    X = df_ml[ALL_FEATURES].copy()
    y = np.log1p(df_ml["num_subscribers"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    feat_df = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, encoders, X_train.columns.tolist(), r2, feat_df


# ===================================================================
# SIDEBAR — DATA FILTERS 
# ===================================================================
def render_sidebar_filters(df: pd.DataFrame):
    """Render data-filtering controls. Returns filtered DataFrame."""
    with st.sidebar:
        st.markdown(
            "<h1 style='text-align: center; color: #2ca02c;'>🎓 Udemy Analytics</h1>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### 🔍 Data Filters")
        st.caption("Filters apply to the Dashboard tab only.")

        # Subject filter
        subject_opts = ["All"] + sorted(df["subject"].dropna().unique().tolist())
        sel_subject = st.selectbox("Course Subject", subject_opts, key="f_subject")

        # Price category filter
        price_opts = ["All"] + sorted(
            df["category_price"].dropna().unique().tolist()
        )
        sel_price = st.selectbox("Price Category", price_opts, key="f_price")

        # Duration category filter
        dur_opts = ["All"] + sorted(
            df["category_duration"].dropna().unique().tolist()
        )
        sel_dur = st.selectbox("Duration Category", dur_opts, key="f_dur")

        st.divider()
        st.caption("💡 The ML model in Tab 2 uses all data.")

        # Apply filters
        filtered = df.copy()
        if sel_subject != "All":
            filtered = filtered[filtered["subject"] == sel_subject]
        if sel_price != "All":
            filtered = filtered[filtered["category_price"] == sel_price]
        if sel_dur != "All":
            filtered = filtered[filtered["category_duration"] == sel_dur]

        return filtered


# ===================================================================
# TAB 1 — BUSINESS DASHBOARD (EDA)
# ===================================================================
def render_dashboard(df: pd.DataFrame):
    """Display KPIs and charts from the filtered data."""

    # ------------------------------------------------------------------
    # 1. KPI METRICS
    # ------------------------------------------------------------------
    st.subheader("📊 Key Metrics")
    total_courses = len(df)
    total_subscribers = int(df["num_subscribers"].sum())
    total_profit = int(df["profit"].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("📚 Total Courses", f"{total_courses:,}")
    m2.metric("👥 Total Subscribers", f"{total_subscribers:,}")
    m3.metric("💰 Total Profit ($)", f"${total_profit:,}")

    st.divider()

    # ------------------------------------------------------------------
    # 2. PIE CHART — Distribution by Subject
    # ------------------------------------------------------------------
    st.subheader("🧩 Course Distribution by Subject")
    subj_counts = df["subject"].value_counts().reset_index()
    subj_counts.columns = ["subject", "count"]
    fig_pie = px.pie(
        subj_counts,
        names="subject",
        values="count",
        color_discrete_sequence=COLOR_SEQUENCE,
        hole=0.40,
        title="Courses by Subject Area",
    )
    fig_pie.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # 3. BAR CHART — Total Subscribers by Level
    # ------------------------------------------------------------------
    st.subheader("📈 Total Subscribers by Course Level")
    subs_by_level = (
        df.groupby("level")["num_subscribers"]
        .sum()
        .reset_index()
        .sort_values("num_subscribers", ascending=False)
    )
    fig_level = px.bar(
        subs_by_level,
        x="level",
        y="num_subscribers",
        color="level",
        color_discrete_sequence=COLOR_SEQUENCE,
        text_auto=".0s",
        title="Total Enrollments per Difficulty Level",
    )
    fig_level.update_layout(
        height=450,
        xaxis_title="",
        yaxis_title="Total Subscribers",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_level, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # 4. LINE CHART — Yearly Profit Growth
    # ------------------------------------------------------------------
    st.subheader("📈 Yearly Revenue Growth")
    profit_by_year = (
        df.groupby("year")["profit"]
        .sum()
        .reset_index()
    )
    fig_line = px.line(
        profit_by_year,
        x="year",
        y="profit",
        markers=True,
        color_discrete_sequence=[PRIMARY_COLOR],
        title="Total Profit Generated per Year",
    )
    fig_line.update_traces(line=dict(width=3), marker=dict(size=8))
    fig_line.update_layout(
        height=450,
        xaxis=dict(dtick=1, tickmode="linear"),
        yaxis_title="Total Profit ($)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # 5. SCATTER PLOT — Price vs Subscribers
    # ------------------------------------------------------------------
    st.subheader("💲 Price vs. Subscribers")
    fig_scatter = px.scatter(
        df,
        x="price",
        y="num_subscribers",
        color="subject",
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.6,
        size="num_reviews",
        hover_data=["num_lectures", "level"],
        trendline="ols",
        title="Course Price vs. Number of Subscribers",
    )
    fig_scatter.update_layout(
        height=500,
        xaxis_title="Price ($)",
        yaxis_title="Subscribers",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ===================================================================
# TAB 2 — PREDICT COURSE SUBSCRIBERS (static ML model)
# ===================================================================
def render_predict_tab(df_full: pd.DataFrame):
    """Prediction interface using the cached static model."""

    # Train (or load from cache)
    model, encoders, feature_order, r2, feat_df = train_static_model()

    # ------------------------------------------------------------------
    # MODEL PERFORMANCE SUMMARY
    # ------------------------------------------------------------------
    st.subheader("🚀 Random Forest — Model Card")
    c1, c2 = st.columns([1, 2])
    c1.metric("🎯 R² Score (log-scale)", f"{r2:.4f}")
    c2.markdown(
        f"**Hyperparameters:** `n_estimators=300` · `max_depth=12` · "
        f"`random_state=42`  \n"
        f"**Features ({len(ALL_FEATURES)}):** "
        f"{', '.join(ALL_FEATURES)}"
    )

    with st.expander("🔥 View Feature Importances"):
        feat_plot = feat_df.sort_values("importance", ascending=True)
        fig_fi = px.bar(
            feat_plot,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=CONTINUOUS_COLORS,
            title="Random Forest Feature Importances",
            text_auto=".3f",
        )
        fig_fi.update_layout(
            height=420,
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # PREDICTION FORM
    # ------------------------------------------------------------------
    st.subheader("🧪 Predict Subscribers for a New Course")

    with st.form("predict_form"):
        st.markdown("##### Enter the course details below")
        col_a, col_b, col_c = st.columns(3)

        # Column A — numeric inputs
        with col_a:
            price = st.slider(
                "💲 Price ($)",
                min_value=0, max_value=200, value=50, step=5,
            )
            num_lectures = st.number_input(
                "📖 Number of Lectures",
                min_value=1, max_value=500, value=30, step=1,
            )
            content_duration = st.slider(
                "⏱ Content Duration (hours)",
                min_value=0.0, max_value=50.0, value=10.0, step=0.5,
            )

        # Column B — numeric inputs
        with col_b:
            num_reviews = st.number_input(
                "⭐ Number of Reviews",
                min_value=0, max_value=20000, value=100, step=10,
            )
            year = st.selectbox(
                "📅 Year",
                options=sorted(df_full["year"].unique().tolist()),
                index=0,
            )
            is_paid = st.selectbox(
                "💳 Is Paid?",
                options=[0, 1],
                format_func=lambda x: "Paid (1)" if x == 1 else "Free (0)",
            )

        # Column C — categorical inputs
        with col_c:
            subject = st.selectbox(
                "📂 Subject",
                options=sorted(df_full["subject"].dropna().unique().tolist()),
            )
            level = st.selectbox(
                "📊 Level",
                options=sorted(df_full["level"].dropna().unique().tolist()),
            )
            duration_cat = st.selectbox(
                "⏳ Duration Category",
                options=sorted(df_full["category_duration"].dropna().unique().tolist()),
            )

        submitted = st.form_submit_button(
            "🔮 Predict Subscribers", type="primary", use_container_width=True,
        )

    if submitted:
        # Build input dict matching training feature order
        input_dict = {
            "price": price,
            "num_lectures": num_lectures,
            "content_duration": content_duration,
            "num_reviews": num_reviews,
            "year": year,
            "is_paid": is_paid,
        }

        # Encode categoricals using the SAME LabelEncoders from training
        cat_mapping = {
            "subject": subject,
            "level": level,
            "category_duration": duration_cat,
        }
        for col, val in cat_mapping.items():
            le = encoders[col]
            try:
                input_dict[col] = le.transform([val])[0]
            except ValueError:
                input_dict[col] = 0  # fallback for unseen category

        # Construct DataFrame in the exact column order used during training
        inp_df = pd.DataFrame([input_dict])
        inp_df = inp_df[feature_order]

        # Predict
        pred_log = model.predict(inp_df)[0]
        pred_actual = np.expm1(pred_log)

        # Display
        st.balloons()
        st.success(
            f"### ✅ Predicted Subscribers: **{pred_actual:,.0f}**"
        )

        with st.expander("📋 View submitted values"):
            st.json(
                {
                    k: (
                        int(v) if isinstance(v, (np.integer, np.int64))
                        else float(v) if isinstance(v, (np.floating, np.float64))
                        else v
                    )
                    for k, v in input_dict.items()
                }
            )


# ===================================================================
# MAIN
# ===================================================================
def main():
    try:
        df_raw = load_data()
    except FileNotFoundError:
        st.error(f"❌ File not found: `{CSV_PATH}`. Please place it in the app directory.")
        st.stop()

    # Sidebar filters (affect Tab 1 only)
    df_filtered = render_sidebar_filters(df_raw)

    # Tabs
    tab1, tab2 = st.tabs(["📈 Business Dashboard", "🔮 Predict Course Subscribers"])

    with tab1:
        render_dashboard(df_filtered)

    with tab2:
        render_predict_tab(df_raw)


if __name__ == "__main__":
    main()