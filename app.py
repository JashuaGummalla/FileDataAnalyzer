"""
File Data Analyzer System - Premium UI/UX Edition
A Python-Based Data Analysis and Visualization System Using NumPy, Pandas, Matplotlib and Seaborn.
"""

import os
import io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Import custom analysis modules
from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.statistics import StatisticalAnalyzer
from modules.visualizations import Visualizer
from modules.insights import InsightsEngine

# Page configuration
st.set_page_config(
    page_title="File Data Analyzer | Enterprise Lab Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Modern UI/UX Design System with Glassmorphism and Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 6px;
    }

    /* Glassmorphism Metric Card */
    .glass-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08);
        border-color: #38bdf8;
    }
    .card-icon {
        font-size: 1.4rem;
        float: right;
        opacity: 0.85;
    }
    .card-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
    }
    .card-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
        line-height: 1.2;
    }
    .card-subtext {
        font-size: 0.78rem;
        color: #0284c7;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Modern Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-blue { background: #e0f2fe; color: #0369a1; }
    .badge-green { background: #dcfce7; color: #15803d; }
    .badge-amber { background: #fef3c7; color: #b45309; }
    .badge-red { background: #fee2e2; color: #b91c1c; }

    /* Interactive Insight Box */
    .insight-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .insight-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    .insight-card.success { border-left-color: #10b981; }
    .insight-card.warning { border-left-color: #f59e0b; }
    .insight-card.danger { border-left-color: #ef4444; }

    /* Streamlit Button Overrides */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
        padding: 0.5rem 1.25rem;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }
    .sidebar-brand {
        padding: 12px 6px;
        border-bottom: 1px solid #334155;
        margin-bottom: 14px;
    }
    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.03em;
    }

    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #475569;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0284c7 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "cleaning_history" not in st.session_state:
        st.session_state.cleaning_history = []
    if "df_history" not in st.session_state:
        st.session_state.df_history = []

init_session_state()


def push_history(df_prev: pd.DataFrame, message: str):
    """Pushes previous DataFrame state to undo stack."""
    st.session_state.df_history.append(df_prev.copy())
    st.session_state.cleaning_history.append(message)


# Sidebar Header and Navigation
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-title">📊 File Data Analyzer</div>
    <div class="sidebar-brand-sub">NUMPY • PANDAS • MATPLOTLIB • SEABORN</div>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation Workspace",
    [
        "📊 Dashboard",
        "📁 Upload Dataset",
        "🔍 Data Explorer",
        "🧹 Data Cleaning",
        "📈 Statistical Analysis",
        "📊 Visualizations",
        "🔗 Correlation Analysis",
        "💡 Insights",
        "📥 Export",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

# Quick Dataset Status in Sidebar
if st.session_state.df is not None:
    st.sidebar.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px;">
        <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Active Dataset</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; word-break: break-all; margin-top: 2px;">{st.session_state.filename}</div>
        <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;">📏 <strong>{len(st.session_state.df):,}</strong> rows &nbsp;|&nbsp; <strong>{len(st.session_state.df.columns)}</strong> cols</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔄 Reset to Original Data", use_container_width=True):
        st.session_state.df = st.session_state.df_raw.copy()
        st.session_state.cleaning_history = ["Reset to original raw dataset."]
        st.session_state.df_history = []
        st.sidebar.toast("Dataset reset successfully!")
        st.rerun()
else:
    st.sidebar.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px;">
        <div style="font-size: 0.82rem; color: #94a3b8;">No dataset active yet. Upload a file or load the sample data below.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("✨ Load Sample Sales Data", type="primary", use_container_width=True):
        sample_df, err = DataLoader.load_sample_dataset()
        if sample_df is not None:
            st.session_state.df_raw = sample_df.copy()
            st.session_state.df = sample_df.copy()
            st.session_state.filename = "sample_sales_data.csv"
            st.session_state.cleaning_history = ["Loaded built-in sample sales dataset."]
            st.session_state.df_history = []
            st.rerun()
        else:
            st.sidebar.error(err)


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if menu == "📊 Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📊 File Data Analyzer Dashboard</h1>
        <div class="hero-subtitle">Interactive data health monitoring, schema exploration, and real-time metric indicators.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("💡 **Welcome to File Data Analyzer!** Please load a dataset to begin exploration.")
        c_init1, c_init2, _ = st.columns([2, 2, 4])
        with c_init1:
            if st.button("✨ Load Built-In Sample Sales Data", type="primary", use_container_width=True):
                sample_df, err = DataLoader.load_sample_dataset()
                if sample_df is not None:
                    st.session_state.df_raw = sample_df.copy()
                    st.session_state.df = sample_df.copy()
                    st.session_state.filename = "sample_sales_data.csv"
                    st.session_state.cleaning_history = ["Loaded sample dataset."]
                    st.rerun()
        with c_init2:
            st.caption("Or navigate to **📁 Upload Dataset** in the sidebar.")
        st.stop()

    df = st.session_state.df
    meta = DataLoader.get_dataset_metadata(df, st.session_state.filename)
    quality = StatisticalAnalyzer.calculate_data_quality_score(df)

    # 6 Top KPI Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <span class="card-icon">📄</span>
            <div class="card-label">Total Records</div>
            <div class="card-value">{meta['rows']:,}</div>
            <div class="card-subtext">Rows Loaded</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <span class="card-icon">🏛️</span>
            <div class="card-label">Attributes</div>
            <div class="card-value">{meta['columns']}</div>
            <div class="card-subtext">Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="glass-card">
            <span class="card-icon">🔢</span>
            <div class="card-label">Numerical</div>
            <div class="card-value">{len(meta['numerical_cols'])}</div>
            <div class="card-subtext">Continuous Vars</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="glass-card">
            <span class="card-icon">🏷️</span>
            <div class="card-label">Categorical</div>
            <div class="card-value">{len(meta['categorical_cols'])}</div>
            <div class="card-subtext">Discrete Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="glass-card">
            <span class="card-icon">🩹</span>
            <div class="card-label">Missing Cells</div>
            <div class="card-value" style="color: {'#ef4444' if meta['missing_values'] > 0 else '#10b981'};">{meta['missing_values']:,}</div>
            <div class="card-subtext">{quality['missing_pct']}% of data volume</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="glass-card">
            <span class="card-icon">👥</span>
            <div class="card-label">Duplicates</div>
            <div class="card-value" style="color: {'#f59e0b' if meta['duplicate_rows'] > 0 else '#10b981'};">{meta['duplicate_rows']:,}</div>
            <div class="card-subtext">{quality['duplicate_pct']}% duplicate rows</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Data Quality Gauge Card
    q_col1, q_col2 = st.columns([1.2, 2.8])
    with q_col1:
        grade_badge_class = "badge-green" if quality["score"] >= 85 else "badge-amber" if quality["score"] >= 70 else "badge-red"
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="card-label">Data Quality Index</div>
                <span class="badge-pill {grade_badge_class}">Grade {quality['grade']}</span>
            </div>
            <div class="card-value" style="font-size: 2.3rem; color: #0284c7; margin-top: 6px;">{quality['score']}%</div>
            <div style="margin-top: 8px;">
                <div style="background: #e2e8f0; border-radius: 9999px; height: 8px; width: 100%; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #38bdf8, #0284c7); height: 100%; width: {quality['score']}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with q_col2:
        st.markdown(f"""
        <div class="glass-card" style="height: 100%;">
            <div class="card-label">Quality Score Breakdown</div>
            <p style="font-size: 0.9rem; color: #334155; margin-top: 8px; line-height: 1.5;">
                {quality['explanation']}
            </p>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <span class="badge-pill badge-blue">Memory: {meta['memory_usage']}</span>
                <span class="badge-pill badge-green">Records: {meta['rows']:,}</span>
                <span class="badge-pill {'badge-amber' if meta['missing_values']>0 else 'badge-green'}">Nulls: {meta['missing_values']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Inspection Tabs
    tab_prev, tab_schema, tab_missing = st.tabs([
        "👁️ Dataset Preview (First 15 Rows)",
        "📋 Schema, Types & Memory Footprint",
        "📉 Missing Value Visualizer"
    ])

    with tab_prev:
        st.dataframe(df.head(15), use_container_width=True)

    with tab_schema:
        schema_df = pd.DataFrame({
            "Column Attribute": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notna().sum(),
            "Null Count": df.isna().sum(),
            "Null Percentage": (df.isna().sum() / len(df) * 100).round(2).astype(str) + "%",
            "Unique Count": df.nunique()
        }).reset_index(drop=True)
        st.dataframe(schema_df, use_container_width=True)
        st.caption(f"Estimated Deep Memory Allocation: **{meta['memory_usage']}**")

    with tab_missing:
        fig_m, has_missing = Visualizer.plot_missing_values(df)
        if has_missing:
            st.pyplot(fig_m)
            plt.close(fig_m)
        else:
            st.success("🎉 Perfect Completeness! There are zero missing values across all attributes in this dataset.")


# ==============================================================================
# PAGE 2: UPLOAD DATASET
# ==============================================================================
elif menu == "📁 Upload Dataset":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📁 Upload & Ingest Dataset</h1>
        <div class="hero-subtitle">Import CSV or Excel files with automated format validation and encoding fallback.</div>
    </div>
    """, unsafe_allow_html=True)

    u_col1, u_col2 = st.columns([1.8, 1.2])

    with u_col1:
        st.markdown("### 📤 File Ingestion")
        uploaded_file = st.file_uploader(
            "Select dataset file (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            help="Multi-encoding engine handles UTF-8, Latin-1, CP1252, and ISO-8859-1 automatically."
        )

        if uploaded_file is not None:
            with st.spinner("Analyzing and parsing file..."):
                df_loaded, err = DataLoader.load_file(uploaded_file, uploaded_file.name)
                if err:
                    st.error(f"❌ {err}")
                elif df_loaded is not None and not df_loaded.empty:
                    st.session_state.df_raw = df_loaded.copy()
                    st.session_state.df = df_loaded.copy()
                    st.session_state.filename = uploaded_file.name
                    st.session_state.cleaning_history = [f"Loaded custom file '{uploaded_file.name}'."]
                    st.session_state.df_history = []
                    st.success(f"✅ Loaded **{uploaded_file.name}** ({len(df_loaded):,} rows × {len(df_loaded.columns)} columns)!")

    with u_col2:
        st.markdown("### 🧪 Built-In Lab Sample")
        st.markdown("""
        Test the analyzer instantly with our curated sales dataset containing realistic business metrics, dates, and intentional anomalies.
        """)
        if st.button("✨ Load Built-In Sample Sales Dataset", type="primary", use_container_width=True):
            sample_df, err = DataLoader.load_sample_dataset()
            if sample_df is not None:
                st.session_state.df_raw = sample_df.copy()
                st.session_state.df = sample_df.copy()
                st.session_state.filename = "sample_sales_data.csv"
                st.session_state.cleaning_history = ["Loaded sample sales dataset."]
                st.session_state.df_history = []
                st.success("✅ Sample sales dataset loaded! Head to **Dashboard** or **Data Explorer**.")
            else:
                st.error(err)


# ==============================================================================
# PAGE 3: DATA EXPLORER
# ==============================================================================
elif menu == "🔍 Data Explorer":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🔍 Interactive Data Explorer</h1>
        <div class="hero-subtitle">Multi-dimensional filtering, custom row slicing, and dynamic full-text search.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        st.stop()

    df = st.session_state.df

    # View Controls Ribbon
    c_view1, c_view2, c_view3 = st.columns([1.5, 2, 2.5])
    with c_view1:
        view_mode = st.selectbox("Row Slicing Mode", ["First N rows", "Last N rows", "Entire Dataset", "Random Sample"])
    with c_view2:
        if view_mode in ["First N rows", "Last N rows", "Random Sample"]:
            n_rows_slice = st.slider("Row Count", min_value=5, max_value=min(len(df), 500), value=20, step=5)
        else:
            n_rows_slice = len(df)
    with c_view3:
        selected_cols = st.multiselect("Visible Columns", options=list(df.columns), default=list(df.columns))

    # Multi-Condition Filtering Expander
    with st.expander("🛠️ Advanced Dynamic Filtering (Categorical, Numerical & Global Text)", expanded=True):
        f_cols = st.columns(3)
        filtered_df = df.copy()

        # Categorical Filter
        cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        if cat_cols:
            with f_cols[0]:
                filter_cat_col = st.selectbox("Filter by Category", ["None"] + cat_cols)
                if filter_cat_col != "None":
                    unique_vals = list(df[filter_cat_col].dropna().unique())
                    selected_vals = st.multiselect(f"Classes in {filter_cat_col}", options=unique_vals, default=unique_vals[:3])
                    if selected_vals:
                        filtered_df = filtered_df[filtered_df[filter_cat_col].isin(selected_vals)]

        # Numerical Range Filter
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if num_cols:
            with f_cols[1]:
                filter_num_col = st.selectbox("Filter by Numerical Range", ["None"] + num_cols)
                if filter_num_col != "None":
                    min_val = float(df[filter_num_col].min())
                    max_val = float(df[filter_num_col].max())
                    if min_val < max_val:
                        num_range = st.slider(f"Range for {filter_num_col}", min_value=min_val, max_value=max_val, value=(min_val, max_val))
                        filtered_df = filtered_df[(filtered_df[filter_num_col] >= num_range[0]) & (filtered_df[filter_num_col] <= num_range[1])]

        # Global Search Filter
        with f_cols[2]:
            search_query = st.text_input("Global Full-Text Search", "", placeholder="e.g. Laptop, Consumer, etc.")
            if search_query:
                mask = np.column_stack([filtered_df[col].astype(str).str.contains(search_query, case=False, na=False) for col in filtered_df.columns])
                filtered_df = filtered_df[mask.any(axis=1)]

    if not selected_cols:
        st.warning("Please select at least one visible column.")
        st.stop()

    display_df = filtered_df[selected_cols]
    if view_mode == "First N rows":
        display_df = display_df.head(n_rows_slice)
    elif view_mode == "Last N rows":
        display_df = display_df.tail(n_rows_slice)
    elif view_mode == "Random Sample":
        display_df = display_df.sample(min(n_rows_slice, len(display_df)))

    st.markdown(f"**Showing {len(display_df):,} records** (Filtered from {len(df):,} total rows)")
    st.dataframe(display_df, use_container_width=True)


# ==============================================================================
# PAGE 4: DATA CLEANING
# ==============================================================================
elif menu == "🧹 Data Cleaning":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🧹 Data Preprocessing Studio</h1>
        <div class="hero-subtitle">Statistical imputation, duplicate deduplication, column standardization, and reversible workflows.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
        st.stop()

    df = st.session_state.df

    # Undo / Reset Ribbon
    col_u1, col_u2, _ = st.columns([1.5, 1.5, 4])
    with col_u1:
        if st.button("↩️ Undo Last Cleaning Step", disabled=len(st.session_state.df_history) == 0, use_container_width=True):
            st.session_state.df = st.session_state.df_history.pop()
            st.session_state.cleaning_history.append("Undid last operation.")
            st.toast("Reverted last cleaning operation!")
            st.rerun()
    with col_u2:
        if st.button("🔄 Reset to Original Raw Data", use_container_width=True):
            st.session_state.df = st.session_state.df_raw.copy()
            st.session_state.cleaning_history = ["Reset to original raw dataset."]
            st.session_state.df_history = []
            st.toast("Reset to raw dataset!")
            st.rerun()

    st.markdown("---")

    clean_tab1, clean_tab2, clean_tab3, clean_tab4, clean_tab5 = st.tabs([
        "1. 🗑️ Remove Duplicates",
        "2. 🩹 Impute Missing Values",
        "3. 🏷️ Rename Column Headers",
        "4. 🔀 Data Type Casting",
        "5. 📜 Cleaning History Log"
    ])

    with clean_tab1:
        dup_count = int(df.duplicated().sum())
        st.markdown(f"### Duplicate Records Detected: `{dup_count}` ({dup_count / len(df) * 100:.2f}%)")
        if dup_count > 0:
            st.dataframe(df[df.duplicated(keep=False)].head(10), use_container_width=True)
            if st.button("🚀 Eliminate All Duplicate Rows", type="primary"):
                prev_df = df.copy()
                cleaned_df, removed = DataCleaner.remove_duplicates(df)
                st.session_state.df = cleaned_df
                push_history(prev_df, f"Eliminated {removed} duplicate rows.")
                st.success(f"✅ Successfully removed {removed} duplicate rows!")
                st.rerun()
        else:
            st.success("✅ Dataset contains zero duplicate rows.")

    with clean_tab2:
        st.markdown("### Missing Value Imputation Engine")
        cols_with_na = df.columns[df.isna().any()].tolist()
        
        if not cols_with_na:
            st.success("✅ No missing values found in current active dataset!")
        else:
            st.info(f"Columns containing missing values: **{', '.join(cols_with_na)}**")
            
            c_mv1, c_mv2 = st.columns(2)
            with c_mv1:
                strategy = st.selectbox(
                    "Choose Imputation / Dropping Strategy",
                    [
                        ("fill_mean", "Fill Numerical with NumPy Mean"),
                        ("fill_median", "Fill Numerical with NumPy Median"),
                        ("fill_mode", "Fill with Most Frequent Value (Mode)"),
                        ("fill_zero", "Fill Numerical with Zero (0)"),
                        ("fill_custom", "Fill with Custom String (e.g. 'Unknown')"),
                        ("drop_rows", "Drop Rows with Missing Values")
                    ],
                    format_func=lambda x: x[1]
                )[0]
                
            with c_mv2:
                target_cols = st.multiselect("Select Target Columns", options=cols_with_na, default=cols_with_na)

            custom_val = None
            if strategy == "fill_custom":
                custom_val = st.text_input("Enter Custom Imputation Value", "Unknown")

            if st.button("⚡ Execute Imputation Strategy", type="primary"):
                prev_df = df.copy()
                cleaned_df, report = DataCleaner.handle_missing_values(df, strategy=strategy, columns=target_cols, custom_fill=custom_val)
                st.session_state.df = cleaned_df
                push_history(prev_df, f"Applied '{strategy}' to {target_cols}. Handled {report['imputed_or_dropped_count']} missing cells.")
                st.success(f"✅ Imputation complete! Remaining nulls: {report['remaining_missing']}")
                st.rerun()

    with clean_tab3:
        st.markdown("### Column Header Formatting")
        c_rn1, c_rn2 = st.columns(2)
        with c_rn1:
            st.markdown("#### Automated Standardization")
            style = st.selectbox("Naming Convention", ["snake_case", "lowercase", "uppercase", "strip_spaces"])
            if st.button("✨ Apply Convention to All Headers"):
                prev_df = df.copy()
                cleaned_df, mapping = DataCleaner.standardize_column_names(df, style=style)
                st.session_state.df = cleaned_df
                push_history(prev_df, f"Standardized headers to {style}.")
                st.success(f"✅ Column names transformed to {style}!")
                st.rerun()

        with c_rn2:
            st.markdown("#### Manual Column Rename")
            old_col = st.selectbox("Target Column", options=list(df.columns))
            new_col = st.text_input("New Name", value=old_col)
            if st.button("✏️ Rename Column"):
                prev_df = df.copy()
                cleaned_df, err = DataCleaner.rename_single_column(df, old_col, new_col)
                if err:
                    st.error(err)
                else:
                    st.session_state.df = cleaned_df
                    push_history(prev_df, f"Renamed column '{old_col}' to '{new_col}'.")
                    st.success(f"✅ Renamed '{old_col}' → '{new_col}'")
                    st.rerun()

    with clean_tab4:
        st.markdown("### Safe Data Type Conversion")
        c_tc1, c_tc2, c_tc3 = st.columns([2, 2, 1.5])
        with c_tc1:
            col_to_cast = st.selectbox("Select Column to Cast", options=list(df.columns))
        with c_tc2:
            target_type = st.selectbox("Target Type", ["integer", "float", "string", "datetime", "category", "boolean"])
        with c_tc3:
            st.write("")
            st.write("")
            if st.button("🔀 Execute Cast", type="primary"):
                prev_df = df.copy()
                cleaned_df, err = DataCleaner.convert_column_type(df, col_to_cast, target_type)
                if err:
                    st.error(err)
                else:
                    st.session_state.df = cleaned_df
                    push_history(prev_df, f"Converted column '{col_to_cast}' to {target_type}.")
                    st.success(f"✅ Converted '{col_to_cast}' to {target_type}!")
                    st.rerun()

    with clean_tab5:
        st.markdown("### Transformation History Log")
        if st.session_state.cleaning_history:
            for idx, item in enumerate(st.session_state.cleaning_history, 1):
                st.markdown(f"**Step {idx}:** `{item}`")
        else:
            st.info("No preprocessing steps applied yet (raw dataset state).")


# ==============================================================================
# PAGE 5: STATISTICAL ANALYSIS
# ==============================================================================
elif menu == "📈 Statistical Analysis":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📈 Statistical Analysis & NumPy Lab</h1>
        <div class="hero-subtitle">Vectorized descriptive statistics, statistical moments, and non-parametric IQR outlier detection.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    num_summary = StatisticalAnalyzer.compute_numerical_summary(df)

    stat_tab1, stat_tab2, stat_tab3, stat_tab4 = st.tabs([
        "📊 Numerical Descriptive Statistics",
        "🔢 NumPy Vectorization Inspector",
        "🏷️ Categorical Frequency Analysis",
        "🎯 IQR Outlier Detection Table"
    ])

    with stat_tab1:
        st.markdown("### Parametric & Non-Parametric Numerical Summary")
        if num_summary.empty:
            st.warning("No continuous numerical variables found in this dataset.")
        else:
            st.dataframe(num_summary.T, use_container_width=True)
            st.caption("Calculated using direct vectorized NumPy array operations (`np.mean`, `np.median`, `np.std`, `np.var`, `np.percentile`, `np.ptp`).")

    with stat_tab2:
        st.markdown("### NumPy Vectorized Execution & Memory Inspector")
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if num_cols:
            selected_np_col = st.selectbox("Select Continuous Feature for NumPy Analysis", options=num_cols)
            np_data = StatisticalAnalyzer.get_column_numpy_inspection(df, selected_np_col)
            
            c_np1, c_np2 = st.columns(2)
            with c_np1:
                st.markdown("#### Computed NumPy Scalars")
                st.json(np_data)
            with c_np2:
                st.markdown("#### Live Python / NumPy Execution Code")
                st.code(f"""
import numpy as np

# Convert Pandas Series directly to contiguous C-array
arr = df['{selected_np_col}'].dropna().to_numpy(dtype=np.float64)

# Vectorized Statistical Computations
mean_val   = np.mean(arr)            # Output: {np_data.get('np.mean()', 0):,.4f}
median_val = np.median(arr)          # Output: {np_data.get('np.median()', 0):,.4f}
std_val    = np.std(arr, ddof=1)     # Output: {np_data.get('np.std()', 0):,.4f}
var_val    = np.var(arr, ddof=1)     # Output: {np_data.get('np.var()', 0):,.4f}
range_val  = np.ptp(arr)             # Output: {np_data.get('np.ptp() (Range)', 0):,.4f}
q25_val    = np.percentile(arr, 25)  # Output: {np_data.get('np.percentile(25)', 0):,.4f}
q75_val    = np.percentile(arr, 75)  # Output: {np_data.get('np.percentile(75)', 0):,.4f}
                """, language="python")

    with stat_tab3:
        st.markdown("### Categorical Class Distribution Table")
        cat_summary = StatisticalAnalyzer.compute_categorical_summary(df)
        if cat_summary.empty:
            st.info("No categorical attributes detected.")
        else:
            st.dataframe(cat_summary, use_container_width=True)

    with stat_tab4:
        st.markdown("### IQR Outlier Detection Pipeline")
        st.markdown(r"""
        Outlier boundaries are determined using the non-parametric **Interquartile Range**:
        $$\text{IQR} = Q_3 - Q_1 \quad \implies \quad [\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}]$$
        """)
        outliers_df = StatisticalAnalyzer.detect_outliers_iqr(df)
        if outliers_df.empty:
            st.info("No numerical features available for outlier detection.")
        else:
            st.dataframe(outliers_df, use_container_width=True)


# ==============================================================================
# PAGE 6: VISUALIZATIONS
# ==============================================================================
elif menu == "📊 Visualizations":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📊 Publication Visualization Studio</h1>
        <div class="hero-subtitle">High-DPI statistical plots crafted with Matplotlib and Seaborn with custom color themes.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    all_cols = list(df.columns)
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    c_cfg1, c_cfg2 = st.columns([1, 2.5])

    with c_cfg1:
        st.markdown("### ⚙️ Chart Configuration")
        chart_type = st.selectbox(
            "Visualization Type",
            [
                "Histogram & KDE",
                "Bar Chart",
                "Line Chart",
                "Scatter Plot",
                "Box & Whisker Plot",
                "Pie / Donut Chart",
                "Seaborn Count Plot",
                "Correlation Heatmap"
            ]
        )
        palette = st.selectbox("Aesthetic Palette", list(Visualizer.COLOR_PALETTES.keys()), index=0)

        # Dynamic Controls per Chart Type
        x_col, y_col, hue_col = None, None, None
        bins, kde, show_trend = 20, True, False

        if chart_type == "Histogram & KDE":
            if not num_cols:
                st.error("No continuous numerical variables for Histogram.")
            else:
                x_col = st.selectbox("Numerical Feature (X)", options=num_cols)
                bins = st.slider("Bin Granularity", 5, 50, 20)
                kde = st.checkbox("Overlay KDE Density Curve", value=True)

        elif chart_type == "Bar Chart":
            x_col = st.selectbox("Grouping Feature (X)", options=all_cols)
            agg_type = st.selectbox("Aggregation Metric", ["Frequency Count", "Sum of Y", "Mean of Y"])
            if agg_type != "Frequency Count":
                y_col = st.selectbox("Numerical Metric (Y)", options=num_cols)
                agg_func = "sum" if "Sum" in agg_type else "mean"
            else:
                y_col = None
                agg_func = "count"

        elif chart_type == "Line Chart":
            x_col = st.selectbox("X-Axis (Time / Sequence)", options=all_cols)
            y_col = st.selectbox("Y-Axis (Numerical Metric)", options=num_cols if num_cols else all_cols)
            hue_opt = ["None"] + cat_cols
            hue_sel = st.selectbox("Hue Grouping (Optional)", options=hue_opt)
            hue_col = None if hue_sel == "None" else hue_sel

        elif chart_type == "Scatter Plot":
            if len(num_cols) < 2:
                st.error("Scatter plots require at least 2 numerical features.")
            else:
                x_col = st.selectbox("Independent Variable (X)", options=num_cols, index=0)
                y_col = st.selectbox("Dependent Variable (Y)", options=num_cols, index=min(1, len(num_cols)-1))
                hue_opt = ["None"] + cat_cols
                hue_sel = st.selectbox("Color Hue (Optional)", options=hue_opt)
                hue_col = None if hue_sel == "None" else hue_sel
                show_trend = st.checkbox("Include Linear Regression Trendline", value=False)

        elif chart_type == "Box & Whisker Plot":
            if not num_cols:
                st.error("Box plot requires numerical variables.")
            else:
                y_col = st.selectbox("Numerical Feature (Y)", options=num_cols)
                x_opt = ["None"] + cat_cols
                x_sel = st.selectbox("Group by Category (X - Optional)", options=x_opt)
                x_col = None if x_sel == "None" else x_sel

        elif chart_type == "Pie / Donut Chart":
            if not cat_cols:
                st.error("Pie charts require categorical features.")
            else:
                x_col = st.selectbox("Categorical Feature", options=cat_cols)
                is_donut = st.checkbox("Donut Hole Style", value=True)
                max_slices = st.slider("Max Slices", 3, 10, 6)

        elif chart_type == "Seaborn Count Plot":
            x_col = st.selectbox("Categorical Feature", options=cat_cols if cat_cols else all_cols)
            hue_opt = ["None"] + cat_cols
            hue_sel = st.selectbox("Group by Hue (Optional)", options=hue_opt)
            hue_col = None if hue_sel == "None" else hue_sel

        elif chart_type == "Correlation Heatmap":
            pass

    with c_cfg2:
        st.markdown("### 🖼️ Rendered Visualization Canvas")
        fig = None

        try:
            if chart_type == "Histogram & KDE" and x_col:
                fig = Visualizer.plot_histogram(df, column=x_col, bins=bins, kde=kde, palette_name=palette)
            elif chart_type == "Bar Chart" and x_col:
                fig = Visualizer.plot_bar_chart(df, x_col=x_col, y_col=y_col, agg_func=agg_func if y_col else "count", palette_name=palette)
            elif chart_type == "Line Chart" and x_col and y_col:
                fig = Visualizer.plot_line_chart(df, x_col=x_col, y_col=y_col, hue_col=hue_col, palette_name=palette)
            elif chart_type == "Scatter Plot" and x_col and y_col:
                fig = Visualizer.plot_scatter(df, x_col=x_col, y_col=y_col, hue_col=hue_col, show_trendline=show_trend, palette_name=palette)
            elif chart_type == "Box & Whisker Plot" and y_col:
                fig = Visualizer.plot_box_plot(df, y_col=y_col, x_col=x_col, palette_name=palette)
            elif chart_type == "Pie / Donut Chart" and x_col:
                fig, err = Visualizer.plot_pie_chart(df, column=x_col, max_slices=max_slices, donut=is_donut, palette_name=palette)
                if err:
                    st.warning(err)
            elif chart_type == "Seaborn Count Plot" and x_col:
                fig = Visualizer.plot_countplot(df, column=x_col, hue_col=hue_col, palette_name=palette)
            elif chart_type == "Correlation Heatmap":
                corr_matrix, _ = StatisticalAnalyzer.compute_correlation_matrix(df)
                if corr_matrix.empty:
                    st.warning("Insufficient continuous variables for correlation heatmap.")
                else:
                    fig = Visualizer.plot_correlation_heatmap(corr_matrix, palette_name=palette)

            if fig is not None:
                st.pyplot(fig)
                
                # High-Res Export Buffer
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", dpi=300, bbox_inches="tight")
                st.download_button(
                    label="💾 Download Publication Chart (High-Res 300 DPI PNG)",
                    data=img_buf.getvalue(),
                    file_name=f"{chart_type.replace(' ', '_').lower()}.png",
                    mime="image/png",
                    type="primary"
                )
                plt.close(fig)

        except Exception as e:
            st.error(f"❌ Error rendering visualization: {str(e)}")


# ==============================================================================
# PAGE 7: CORRELATION ANALYSIS
# ==============================================================================
elif menu == "🔗 Correlation Analysis":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🔗 Correlation Matrix & Feature Collinearity</h1>
        <div class="hero-subtitle">Bivariate dependency analysis, collinearity ranking, and statistical commentary.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    corr_matrix, strong_pairs = StatisticalAnalyzer.compute_correlation_matrix(df)

    if corr_matrix.empty:
        st.warning("At least two numerical features are required for correlation analysis.")
        st.stop()

    st.markdown("""
    <div class="insight-card">
        <strong>⚠️ Statistical Disclaimer:</strong> Pearson correlation coefficient (r) measures linear co-movement (-1.0 to +1.0) and does <strong>not</strong> imply causal relationships due to latent confounding factors.
    </div>
    """, unsafe_allow_html=True)

    c_cr1, c_cr2 = st.columns([1.5, 1.2])

    with c_cr1:
        st.markdown("### Annotated Correlation Heatmap")
        fig_heat = Visualizer.plot_correlation_heatmap(corr_matrix)
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    with c_cr2:
        st.markdown("### Top Feature Relationships")
        if strong_pairs:
            for pair in strong_pairs[:6]:
                val = pair["Correlation"]
                card_class = "success" if val > 0 else "danger"
                color = "#16a34a" if val > 0 else "#dc2626"
                st.markdown(f"""
                <div class="insight-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{pair['Feature 1']} ↔ {pair['Feature 2']}</strong>
                        <span style="font-weight: 800; color: {color};">r = {val:.3f}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">{pair['Strength']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong correlations (|r| >= 0.5) identified.")

    st.markdown("### Complete Correlation Coefficient Matrix Table")
    st.dataframe(corr_matrix.round(4), use_container_width=True)


# ==============================================================================
# PAGE 8: INSIGHTS
# ==============================================================================
elif menu == "💡 Insights":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">💡 Automated Insights Engine</h1>
        <div class="hero-subtitle">Deterministic rule-based pattern extraction, distribution audits, and class balance discovery.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    all_insights = InsightsEngine.generate_all_insights(df)

    ins_tab1, ins_tab2, ins_tab3, ins_tab4, ins_tab5 = st.tabs([
        "🏥 Dataset Health & Hygiene",
        "📈 Numerical & Dispersion Findings",
        "🔗 Correlation Relationships",
        "🏷️ Categorical Balances",
        "🎯 Outlier & Time-Series Alerts"
    ])

    def render_insights_cards(items):
        if not items:
            st.info("No specific observations in this category.")
            return
        for item in items:
            t = item.get("type", "info")
            cls_name = "success" if t == "success" else "warning" if t == "warning" else ""
            title = item.get("title", "Analytical Observation")
            st.markdown(f"""
            <div class="insight-card {cls_name}">
                <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">{title}</div>
                <div style="color: #334155; font-size: 0.9rem; margin-top: 4px; line-height: 1.5;">{item['text']}</div>
            </div>
            """, unsafe_allow_html=True)

    with ins_tab1:
        render_insights_cards(all_insights["health"])

    with ins_tab2:
        render_insights_cards(all_insights["numerical"])

    with ins_tab3:
        render_insights_cards(all_insights["correlation"])

    with ins_tab4:
        render_insights_cards(all_insights["categorical"])

    with ins_tab5:
        st.markdown("#### Outlier Alerts")
        render_insights_cards(all_insights["outliers"])
        st.markdown("#### Temporal / Chronological Findings")
        render_insights_cards(all_insights["time_series"])


# ==============================================================================
# PAGE 9: EXPORT
# ==============================================================================
elif menu == "📥 Export":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📥 Data Export & Report Center</h1>
        <div class="hero-subtitle">Download clean datasets, statistical summaries, correlation matrices, and full analysis reports.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
        st.stop()

    df = st.session_state.df

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-label">1. Cleaned Dataset (CSV)</div>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                Export the currently active preprocessed DataFrame after all deduplication and imputation operations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name=f"cleaned_{st.session_state.filename or 'dataset.csv'}",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="card-label">2. Statistical Summary Table</div>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                Export all computed descriptive statistics (Mean, Median, Std, Range, Percentiles, Skewness).
            </p>
        </div>
        """, unsafe_allow_html=True)
        num_summary = StatisticalAnalyzer.compute_numerical_summary(df)
        if not num_summary.empty:
            stat_csv = num_summary.to_csv().encode("utf-8")
            st.download_button(
                label="📥 Download Statistical Summary (CSV)",
                data=stat_csv,
                file_name="statistical_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No numerical summary available.")

    with exp_col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-label">3. Correlation Matrix (CSV)</div>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                Export the complete Pearson correlation coefficient matrix table.
            </p>
        </div>
        """, unsafe_allow_html=True)
        corr_matrix, _ = StatisticalAnalyzer.compute_correlation_matrix(df)
        if not corr_matrix.empty:
            corr_csv = corr_matrix.to_csv().encode("utf-8")
            st.download_button(
                label="📥 Download Correlation Matrix (CSV)",
                data=corr_csv,
                file_name="correlation_matrix.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No correlation matrix available.")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div class="card-label">4. Comprehensive Analysis Report (.txt)</div>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                Download a fully structured textual report containing metadata, statistics, cleaning logs, and rule-based insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        full_report_text = InsightsEngine.generate_full_text_report(
            df=df,
            dataset_name=st.session_state.filename or "Uploaded Dataset",
            cleaning_logs=st.session_state.cleaning_history
        )
        st.download_button(
            label="📥 Download Comprehensive Report (TXT)",
            data=full_report_text.encode("utf-8"),
            file_name="comprehensive_data_analysis_report.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary"
        )

    st.markdown("---")
    with st.expander("👁️ Live Preview of Analysis Report", expanded=False):
        st.text(full_report_text)


# ==============================================================================
# PAGE 10: ABOUT & VIVA GUIDE
# ==============================================================================
elif menu == "ℹ️ About":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">ℹ️ About File Data Analyzer System</h1>
        <div class="hero-subtitle">Educational platform showcasing practical Python data science workflows for laboratory evaluation.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technology Stack & Library Synergy")

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a;">🐼 Pandas</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                • Tabular I/O (CSV/Excel)<br/>
                • DataFrame slicing & sorting<br/>
                • GroupBy aggregations<br/>
                • Missing value detection<br/>
                • Schema & memory tracking
            </div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a;">🔢 NumPy</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                • C-vectorized execution<br/>
                • <code>np.mean()</code>, <code>np.median()</code><br/>
                • <code>np.std()</code>, <code>np.var()</code><br/>
                • <code>np.percentile()</code>, <code>np.ptp()</code><br/>
                • IQR outlier boundaries
            </div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a;">📊 Matplotlib</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                • High-DPI canvas engine<br/>
                • Figure layout geometry<br/>
                • Custom axes & spines<br/>
                • Threshold indicator lines<br/>
                • 300 DPI image buffers
            </div>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a;">🎨 Seaborn</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 6px;">
                • Distribution KDE plots<br/>
                • Curated color palettes<br/>
                • Correlation heatmaps<br/>
                • Categorical count plots<br/>
                • Regression trendlines
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🎓 Lab Viva & Viva Examination Questions")

    st.markdown("""
    <div class="insight-card">
        <strong>Q1: Why is NumPy used alongside Pandas in this project?</strong><br/>
        <em>Answer:</em> Pandas provides tabular indexing, label-based manipulation, and I/O. NumPy executes low-level, high-performance vectorized calculations on raw contiguous memory buffers without Python interpreter loop overhead.
    </div>

    <div class="insight-card">
        <strong>Q2: How does the Interquartile Range (IQR) method detect outliers?</strong><br/>
        <em>Answer:</em> IQR measures the spread of the middle 50% of the dataset: $IQR = Q_3 - Q_1$. Points below $Q_1 - 1.5 \times IQR$ or above $Q_3 + 1.5 \times IQR$ are flagged as statistical anomalies. This method is non-parametric and robust against skewed distributions.
    </div>

    <div class="insight-card">
        <strong>Q3: How is the Data Quality Score calculated?</strong><br/>
        <em>Answer:</em> The application calculates a penalized quality index:
        $$\\text{Score} = 100 - (\\text{Missing \\%} \\times 0.6 + \\text{Duplicate \\%} \\times 0.4)$$
        It rewards data completeness and record uniqueness.
    </div>
    """, unsafe_allow_html=True)
