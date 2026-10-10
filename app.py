"""
File Data Analyzer System - Modern Analytics Pro Edition
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
    page_title="File Data Analyzer | Enterprise Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Analytics Pro Design System with High-Contrast Color Variables
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg-main: #F4F7FB;
        --card-bg: #FFFFFF;
        --text-primary: #172033;
        --text-secondary: #5B6475;
        --accent-blue: #2563EB;
        --accent-cyan: #0891B2;
        --navy-dark: #14213D;
        --border-color: #E2E8F0;
        --hover-bg: #EFF6FF;
        --color-success: #15803D;
        --color-warning: #B45309;
        --color-danger: #B91C1C;
    }

    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #172033;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1440px;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #14213D 55%, #1E3A8A 100%);
        border: 1px solid rgba(37, 99, 235, 0.3);
        border-radius: 16px;
        padding: 24px 32px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(20, 33, 61, 0.25), 0 4px 10px -2px rgba(20, 33, 61, 0.15);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -10%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.18) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 60%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.25;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #CBD5E1;
        font-weight: 400;
        margin-top: 6px;
        letter-spacing: 0.01em;
    }
    .hero-meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
        align-items: center;
    }

    /* Interactive KPI Cards */
    .kpi-card-wrapper {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #2563EB;
        border-radius: 14px;
        padding: 18px 20px 14px 20px;
        box-shadow: 0 2px 8px rgba(20, 33, 61, 0.04), 0 1px 3px rgba(20, 33, 61, 0.06);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 136px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card-wrapper:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -3px rgba(37, 99, 235, 0.12), 0 4px 6px -2px rgba(20, 33, 61, 0.04);
        border-color: #93C5FD;
    }
    .kpi-card-wrapper.accent-blue { border-top-color: #2563EB; }
    .kpi-card-wrapper.accent-cyan { border-top-color: #0891B2; }
    .kpi-card-wrapper.accent-purple { border-top-color: #7C3AED; }
    .kpi-card-wrapper.accent-amber { border-top-color: #D97706; }
    .kpi-card-wrapper.accent-red { border-top-color: #DC2626; }
    .kpi-card-wrapper.accent-green { border-top-color: #16A34A; }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #5B6475;
    }
    .kpi-icon {
        font-size: 1.35rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: #F1F5F9;
    }
    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #172033;
        margin-top: 6px;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #2563EB;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    .badge-navy { background: rgba(15, 23, 42, 0.85); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.35); }
    .badge-blue { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .badge-green { background: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }
    .badge-amber { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-red { background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }

    /* Interactive Insight Cards */
    .insight-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(20, 33, 61, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .insight-card:hover {
        transform: translateX(3px);
        box-shadow: 0 4px 12px rgba(20, 33, 61, 0.06);
    }
    .insight-card.success { border-left-color: #16A34A; }
    .insight-card.warning { border-left-color: #D97706; }
    .insight-card.danger { border-left-color: #DC2626; }

    /* Clean Content Panels */
    .glass-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(20, 33, 61, 0.04);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
        padding: 0.45rem 1.1rem;
        border: 1px solid #CBD5E1;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }

    /* Popover button tweaks */
    div[data-testid="stPopover"] > button {
        border-radius: 8px;
        font-size: 0.8rem !important;
        padding: 4px 10px !important;
        font-weight: 600 !important;
        border: 1px solid #CBD5E1 !important;
        background: #F8FAFC !important;
        color: #334155 !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background: #EFF6FF !important;
        color: #2563EB !important;
        border-color: #93C5FD !important;
    }

    /* Popover Content Container Styling */
    div[data-testid="stPopoverBody"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 25px -5px rgba(20, 33, 61, 0.15) !important;
        padding: 16px !important;
    }
    div[data-testid="stPopoverBody"] p,
    div[data-testid="stPopoverBody"] span,
    div[data-testid="stPopoverBody"] label {
        color: #172033 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B132B 0%, #14213D 60%, #1E293B 100%) !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #F1F5F9 !important;
    }
    .sidebar-brand {
        padding: 14px 6px;
        border-bottom: 1px solid rgba(51, 65, 85, 0.8);
        margin-bottom: 16px;
    }
    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.01em;
    }
    section[data-testid="stSidebar"] .sidebar-brand-sub {
    font-size: 0.75rem;
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-top: 2px;
}

    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
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
   .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 2px 6px rgba(20, 33, 61, 0.08);
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
    if "nav_selection" not in st.session_state:
        st.session_state.nav_selection = "📊 Dashboard"
    if "explorer_target" not in st.session_state:
        st.session_state.explorer_target = "all"

init_session_state()


def push_history(df_prev: pd.DataFrame, message: str):
    """Pushes previous DataFrame state to undo stack."""
    st.session_state.df_history.append(df_prev.copy())
    st.session_state.cleaning_history.append(message)


def set_nav(page_name: str, explorer_target: str = "all"):
    """
    Standard on_click callback that safely modifies session state
    BEFORE widgets are instantiated during the subsequent rerun.
    """
    st.session_state["nav_selection"] = page_name
    st.session_state["explorer_target"] = explorer_target


# Sidebar Header and Navigation
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-title">📊 File Data Analyzer</div>
    <div class="sidebar-brand-sub">ENTERPRISE ANALYTICS LAB</div>
</div>
""", unsafe_allow_html=True)

nav_options = [
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

menu = st.sidebar.radio(
    "Navigation Workspace",
    nav_options,
    key="nav_selection"
)

st.sidebar.markdown("---")

# Quick Dataset Status in Sidebar
if st.session_state.df is not None:
    st.sidebar.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;">
        <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Active Dataset</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; word-break: break-all; margin-top: 3px;">{st.session_state.filename}</div>
        <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 6px;">📏 <strong>{len(st.session_state.df):,}</strong> rows &nbsp;|&nbsp; <strong>{len(st.session_state.df.columns)}</strong> cols</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        if st.sidebar.button("🔄 Reset Raw", key="sb_reset_raw", use_container_width=True, help="Revert all cleaning operations back to raw upload."):
            st.session_state.df = st.session_state.df_raw.copy()
            st.session_state.cleaning_history = ["Reset to original raw dataset."]
            st.session_state.df_history = []
            st.sidebar.toast("Dataset reset to original raw state!")
            st.rerun()
    with col_sb2:
        st.sidebar.button("🔍 Explorer", key="sb_goto_explorer", use_container_width=True, on_click=set_nav, args=("🔍 Data Explorer", "all"), help="Jump straight to interactive Data Explorer")
else:
    st.sidebar.markdown("""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;">
        <div style="font-size: 0.82rem; color: #94a3b8;">No dataset active yet. Upload a file or load the sample data below.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("✨ Load Sample Sales Data", key="sb_load_sample", type="primary", use_container_width=True):
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
    # Hero Banner
    filename_badge = f"""<span class="badge-pill badge-navy">📁 {st.session_state.filename}</span>""" if st.session_state.filename else ""
    st.markdown(f"""
    <div class="hero-banner">
        <h1 class="hero-title">📊 File Data Analyzer Dashboard</h1>
        <div class="hero-subtitle">Interactive data health monitoring, schema exploration, and real-time metric indicators.</div>
      
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("💡 **Welcome to File Data Analyzer!** Please load a dataset to begin exploration.")
        c_init1, c_init2, _ = st.columns([2, 2, 4])
        with c_init1:
            if st.button("✨ Load Built-In Sample Sales Data", key="dash_load_sample_hero", type="primary", use_container_width=True):
                sample_df, err = DataLoader.load_sample_dataset()
                if sample_df is not None:
                    st.session_state.df_raw = sample_df.copy()
                    st.session_state.df = sample_df.copy()
                    st.session_state.filename = "sample_sales_data.csv"
                    st.session_state.cleaning_history = ["Loaded sample dataset."]
                    st.rerun()
        with c_init2:
            st.button("📁 Go to File Uploader ➔", key="dash_goto_upload", on_click=set_nav, args=("📁 Upload Dataset", "all"), use_container_width=True)
        st.stop()

    df = st.session_state.df
    meta = DataLoader.get_dataset_metadata(df, st.session_state.filename)
    quality = StatisticalAnalyzer.calculate_data_quality_score(df)

    st.markdown("### 📌 Interactive Key Performance Indicators (KPIs)")
    st.caption("Hover or open card popovers for compact data previews. Click **'Explore Data ➔'** to jump directly to filtered Data Explorer views.")

    # 6 KPI Cards in a balanced 3-column layout (2 rows of 3 cards)
    # ROW 1
    r1_c1, r1_c2, r1_c3 = st.columns(3)

    # 1. Total Records
    with r1_c1:
        st.markdown(f"""
        <div class="kpi-card-wrapper accent-blue">
            <div>
                <div class="kpi-header">
                    <span class="kpi-label">Total Records</span>
                    <span class="kpi-icon">📄</span>
                </div>
                <div class="kpi-value">{meta['rows']:,}</div>
                <div class="kpi-subtext">Rows Loaded • {meta['file_size']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns([1.1, 1.4])
        with c_act1:
            with st.popover("🔍 Inspect", help="View preview of total records and dataset scale", use_container_width=True):
                st.markdown(f"#### 📄 Dataset Records Overview")
                st.markdown(f"- **Filename:** `{st.session_state.filename}`\n- **Total Active Records:** `{meta['rows']:,}` rows\n- **Estimated Memory Footprint:** `{meta['memory_usage']}`")
                st.markdown("**First 5 Records Preview:**")
                st.dataframe(df.head(5), use_container_width=True)
                st.caption(f"Total of {meta['rows']:,} records available in the full dataset.")
                st.button("Open in Data Explorer ➔", key="pop_btn_records", on_click=set_nav, args=("🔍 Data Explorer", "records"), use_container_width=True)
        with c_act2:
            st.button("Explore Data ➔", key="kpi_btn_records", on_click=set_nav, args=("🔍 Data Explorer", "records"), use_container_width=True, help="Drill down into Full Dataset Table")

    # 2. Attributes / Columns
    with r1_c2:
        st.markdown(f"""
        <div class="kpi-card-wrapper accent-cyan">
            <div>
                <div class="kpi-header">
                    <span class="kpi-label">Attributes / Columns</span>
                    <span class="kpi-icon">🏛️</span>
                </div>
                <div class="kpi-value">{meta['columns']}</div>
                <div class="kpi-subtext">{len(meta['numerical_cols'])} Numerical • {len(meta['categorical_cols'])} Categorical</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns([1.1, 1.4])
        with c_act1:
            with st.popover("🔍 Inspect", help="View column names and schema summary", use_container_width=True):
                st.markdown(f"#### 🏛️ Dataset Schema & Attributes")
                st.markdown(f"- **Total Attributes:** `{meta['columns']}` columns\n- **Numerical:** `{len(meta['numerical_cols'])}`\n- **Categorical/Text:** `{len(meta['categorical_cols'])}`\n- **DateTime:** `{len(meta['datetime_cols'])}`")
                st.markdown("**Column Names & Types:**")
                col_types_df = pd.DataFrame({
                    "Attribute": df.columns,
                    "Type": df.dtypes.astype(str),
                    "Non-Null Count": df.notna().sum()
                }).head(8)
                st.dataframe(col_types_df, use_container_width=True)
                if meta['columns'] > 8:
                    st.caption(f"... and {meta['columns'] - 8} more columns in Data Explorer.")
                st.button("Open Schema Explorer ➔", key="pop_btn_schema", on_click=set_nav, args=("🔍 Data Explorer", "schema"), use_container_width=True)
        with c_act2:
            st.button("Explore Schema ➔", key="kpi_btn_schema", on_click=set_nav, args=("🔍 Data Explorer", "schema"), use_container_width=True, help="Drill down into Column Schema & Types")

    # 3. Numerical Columns
    with r1_c3:
        st.markdown(f"""
        <div class="kpi-card-wrapper accent-purple">
            <div>
                <div class="kpi-header">
                    <span class="kpi-label">Numerical Columns</span>
                    <span class="kpi-icon">🔢</span>
                </div>
                <div class="kpi-value">{len(meta['numerical_cols'])}</div>
                <div class="kpi-subtext">Continuous Variables</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns([1.1, 1.4])
        with c_act1:
            with st.popover("🔍 Inspect", help="View numerical summary metrics", use_container_width=True):
                st.markdown(f"#### 🔢 Numerical Variables ({len(meta['numerical_cols'])})")
                if meta['numerical_cols']:
                    num_summary_quick = df[meta['numerical_cols']].describe().T[['mean', 'min', 'max', 'std']].round(2)
                    st.dataframe(num_summary_quick, use_container_width=True)
                else:
                    st.info("No continuous numerical variables detected.")
                st.button("Explore Numerical ➔", key="pop_btn_num", on_click=set_nav, args=("🔍 Data Explorer", "numerical"), use_container_width=True)
        with c_act2:
            st.button("Explore Num ➔", key="kpi_btn_num", on_click=set_nav, args=("🔍 Data Explorer", "numerical"), use_container_width=True, help="Drill down into Numerical Columns")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ROW 2
    r2_c1, r2_c2, r2_c3 = st.columns(3)

    # 4. Categorical Columns
    with r2_c1:
        st.markdown(f"""
        <div class="kpi-card-wrapper accent-blue">
            <div>
                <div class="kpi-header">
                    <span class="kpi-label">Categorical Columns</span>
                    <span class="kpi-icon">🏷️</span>
                </div>
                <div class="kpi-value">{len(meta['categorical_cols'])}</div>
                <div class="kpi-subtext">Discrete Classes / Categories</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns([1.1, 1.4])
        with c_act1:
            with st.popover("🔍 Inspect", help="View categorical classes and sample values", use_container_width=True):
                st.markdown(f"#### 🏷️ Categorical Variables ({len(meta['categorical_cols'])})")
                if meta['categorical_cols']:
                    cat_samples = []
                    for c in meta['categorical_cols'][:6]:
                        distinct_cnt = df[c].nunique()
                        top_vals = ", ".join([str(x) for x in df[c].dropna().unique()[:3]])
                        cat_samples.append({"Attribute": c, "Unique Classes": distinct_cnt, "Sample Classes": top_vals})
                    st.dataframe(pd.DataFrame(cat_samples), use_container_width=True)
                else:
                    st.info("No categorical features detected.")
                st.button("Explore Categorical ➔", key="pop_btn_cat", on_click=set_nav, args=("🔍 Data Explorer", "categorical"), use_container_width=True)
        with c_act2:
            st.button("Explore Cat ➔", key="kpi_btn_cat", on_click=set_nav, args=("🔍 Data Explorer", "categorical"), use_container_width=True, help="Drill down into Categorical Columns")

    # 5. Missing Cells
    with r2_c2:
        missing_color = "#DC2626" if meta['missing_values'] > 0 else "#16A34A"
        card_accent = "accent-red" if meta['missing_values'] > 0 else "accent-green"
        st.markdown(f"""
        <div class="kpi-card-wrapper {card_accent}">
            <div>
                <div class="kpi-header">
                    <span class="kpi-label">Missing Cells</span>
                    <span class="kpi-icon">🩹</span>
                </div>
                <div class="kpi-value" style="color: {missing_color};">{meta['missing_values']:,}</div>
                <div class="kpi-subtext">{quality['missing_pct']}% of total data volume</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns([1.1, 1.4])
        with c_act1:
            with st.popover("🔍 Inspect", help="View missing value distribution across columns", use_container_width=True):
                st.markdown(f"#### 🩹 Missing Values Audit")
                st.markdown(f"- **Total Missing Cells:** `{meta['missing_values']:,}`\n- **Missing Volume Share:** `{quality['missing_pct']}%`")
                if meta['missing_values'] > 0:
                    missing_breakdown = df.isna().sum()
                    missing_breakdown = missing_breakdown[missing_breakdown > 0].reset_index()
                    missing_breakdown.columns = ["Column", "Null Count"]
                    missing_breakdown["Null %"] = (missing_breakdown["Null Count"] / len(df) * 100).round(2).astype(str) + "%"
                    st.dataframe(missing_breakdown, use_container_width=True)
                else:
                    st.success("🎉 Zero missing values detected across all columns!")
                st.button("Explore Missing Records ➔", key="pop_btn_missing", on_click=set_nav, args=("🔍 Data Explorer", "missing"), use_container_width=True)
        with c_act2:
            st.button("Explore Missing ➔", key="kpi_btn_missing", on_click=set_nav, args=("🔍 Data Explorer", "missing"), use_container_width=True, help="Drill down to rows with missing values")

    # 6. Duplicate Rows
    with r2_c3:
        dup_color = "#D97706" if meta['duplicate_rows'] > 0 else "#16A34A"
        card_accent = "accent-amber" if meta['duplicate_rows'] > 0 else "accent-green"
        st.markdown(f"""
        <div class="kpi-card-wrapper {card_accent}">
            <div>
                <div class="kpi-header">
                    <span class="kpi-label">Duplicate Rows</span>
                    <span class="kpi-icon">👥</span>
                </div>
                <div class="kpi-value" style="color: {dup_color};">{meta['duplicate_rows']:,}</div>
                <div class="kpi-subtext">{quality['duplicate_pct']}% duplicate records</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_act1, c_act2 = st.columns([1.1, 1.4])
        with c_act1:
            with st.popover("🔍 Inspect", help="View duplicate record previews", use_container_width=True):
                st.markdown(f"#### 👥 Duplicate Records Audit")
                st.markdown(f"- **Total Duplicate Rows:** `{meta['duplicate_rows']:,}`\n- **Duplicate Percentage:** `{quality['duplicate_pct']}%`")
                if meta['duplicate_rows'] > 0:
                    st.markdown("**Sample Duplicate Rows:**")
                    st.dataframe(df[df.duplicated(keep=False)].head(5), use_container_width=True)
                else:
                    st.success("🎉 All records in this dataset are distinct and unique!")
                st.button("Explore Duplicate Rows ➔", key="pop_btn_duplicates", on_click=set_nav, args=("🔍 Data Explorer", "duplicates"), use_container_width=True)
        with c_act2:
            st.button("Explore Duplicates ➔", key="kpi_btn_duplicates", on_click=set_nav, args=("🔍 Data Explorer", "duplicates"), use_container_width=True, help="Drill down to duplicate rows")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Data Quality Gauge and Breakdown Section
    q_col1, q_col2 = st.columns([1.3, 2.7])
    with q_col1:
        grade_badge_class = "badge-green" if quality["score"] >= 85 else "badge-amber" if quality["score"] >= 70 else "badge-red"
        bar_gradient = "linear-gradient(90deg, #16A34A, #15803D)" if quality["score"] >= 85 else "linear-gradient(90deg, #D97706, #B45309)" if quality["score"] >= 70 else "linear-gradient(90deg, #DC2626, #B91C1C)"
        st.markdown(f"""
        <div class="glass-panel" style="height: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="kpi-label">Data Quality Index</div>
                <span class="badge-pill {grade_badge_class}">Grade {quality['grade']}</span>
            </div>
            <div class="kpi-value" style="font-size: 2.5rem; color: #2563EB; margin-top: 8px;">{quality['score']}%</div>
            <div style="margin-top: 12px;">
                <div style="background: #E2E8F0; border-radius: 9999px; height: 10px; width: 100%; overflow: hidden;">
                    <div style="background: {bar_gradient}; height: 100%; width: {quality['score']}%; transition: width 0.5s ease;"></div>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #5B6475; margin-top: 10px;">
                Penalized index reflecting data completeness & uniqueness.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with q_col2:
        st.markdown(f"""
        <div class="glass-panel" style="height: 100%;">
            <div class="kpi-label">Quality Score Breakdown & Formula Verification</div>
            <p style="font-size: 0.92rem; color: #172033; margin-top: 10px; line-height: 1.6;">
                {quality['explanation']}
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;">
                <span class="badge-pill badge-blue">💾 Memory: {meta['memory_usage']}</span>
                <span class="badge-pill badge-green">📄 Active Rows: {meta['rows']:,}</span>
                <span class="badge-pill {'badge-red' if meta['missing_values']>0 else 'badge-green'}">🩹 Missing: {meta['missing_values']:,} ({quality['missing_pct']}%)</span>
                <span class="badge-pill {'badge-amber' if meta['duplicate_rows']>0 else 'badge-green'}">👥 Duplicates: {meta['duplicate_rows']:,} ({quality['duplicate_pct']}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Additional Dashboard Visuals: Column Composition & Missing Distribution
    v_col1, v_col2 = st.columns([1.2, 1.8])
    with v_col1:
        st.markdown("#### 🧩 Column Type Composition")
        fig_donut = Visualizer.plot_column_distribution_donut(meta)
        st.pyplot(fig_donut)
        plt.close(fig_donut)

    with v_col2:
        st.markdown("#### 📉 Incomplete Columns & Missing Values")
        fig_m, has_missing = Visualizer.plot_missing_values(df)
        if has_missing:
            st.pyplot(fig_m)
            plt.close(fig_m)
        else:
            st.markdown("""
            <div class="insight-card success" style="margin-top: 24px;">
                <div style="font-weight: 700; color: #15803D; font-size: 1rem;">🎉 100% Data Completeness</div>
                <div style="color: #172033; font-size: 0.9rem; margin-top: 6px;">
                    Zero missing values detected across all columns. No data imputation is required for this dataset.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Inspection Tabs
    tab_prev, tab_schema, tab_actions = st.tabs([
        "👁️ Quick Dataset Preview (First 15 Rows)",
        "📋 Schema, Types & Memory Footprint",
        "🚀 Recommended Next Actions"
    ])

    with tab_prev:
        st.dataframe(df.head(15), use_container_width=True)
        st.caption(f"Showing first 15 of {len(df):,} records. Visit **🔍 Data Explorer** for comprehensive search, filtering, and slicing.")

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

    with tab_actions:
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            st.markdown("""
            <div class="insight-card">
                <strong>🧹 Clean Dataset</strong><br/>
                <span style="font-size: 0.85rem; color: #5B6475;">Eliminate duplicates and impute missing values with NumPy.</span>
            </div>
            """, unsafe_allow_html=True)
            st.button("Go to Data Cleaning ➔", key="btn_act_clean", on_click=set_nav, args=("🧹 Data Cleaning", "all"), use_container_width=True)
        with a_col2:
            st.markdown("""
            <div class="insight-card">
                <strong>📈 Statistical Analysis</strong><br/>
                <span style="font-size: 0.85rem; color: #5B6475;">Vectorized moments, IQR outliers, and NumPy array inspections.</span>
            </div>
            """, unsafe_allow_html=True)
            st.button("Go to Statistics ➔", key="btn_act_stat", on_click=set_nav, args=("📈 Statistical Analysis", "all"), use_container_width=True)
        with a_col3:
            st.markdown("""
            <div class="insight-card">
                <strong>📊 Visualizations</strong><br/>
                <span style="font-size: 0.85rem; color: #5B6475;">Generate 8 publication-grade charts with Matplotlib & Seaborn.</span>
            </div>
            """, unsafe_allow_html=True)
            st.button("Go to Visualizations ➔", key="btn_act_vis", on_click=set_nav, args=("📊 Visualizations", "all"), use_container_width=True)


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
                    
                    c_up_a1, c_up_a2 = st.columns(2)
                    with c_up_a1:
                        st.button("📊 View in Dashboard ➔", key="btn_up_dash", on_click=set_nav, args=("📊 Dashboard", "all"), type="primary", use_container_width=True)
                    with c_up_a2:
                        st.button("🔍 Explore in Data Explorer ➔", key="btn_up_exp", on_click=set_nav, args=("🔍 Data Explorer", "all"), use_container_width=True)

    with u_col2:
        st.markdown("### 🧪 Built-In Lab Sample")
        st.markdown("""
        Test the analyzer instantly with our curated sales dataset containing realistic business metrics, dates, intentional nulls, and duplicate anomalies.
        """)
        if st.button("✨ Load Built-In Sample Sales Dataset", key="btn_up_load_sample", type="primary", use_container_width=True):
            sample_df, err = DataLoader.load_sample_dataset()
            if sample_df is not None:
                st.session_state.df_raw = sample_df.copy()
                st.session_state.df = sample_df.copy()
                st.session_state.filename = "sample_sales_data.csv"
                st.session_state.cleaning_history = ["Loaded sample sales dataset."]
                st.session_state.df_history = []
                st.success("✅ Sample sales dataset loaded! Head to **Dashboard** or **Data Explorer**.")
                st.rerun()
            else:
                st.error(err)


# ==============================================================================
# PAGE 3: DATA EXPLORER
# ==============================================================================
elif menu == "🔍 Data Explorer":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🔍 Interactive Data Explorer Workspace</h1>
        <div class="hero-subtitle">Multi-dimensional filtering, custom row slicing, schema exploration, and dynamic drill-downs.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file or load the sample dataset first.")
        st.stop()

    df = st.session_state.df
    meta = DataLoader.get_dataset_metadata(df, st.session_state.filename)

    # Active Explorer Destination Tab Mapping
    target = st.session_state.get("explorer_target", "all")

    target_descriptions = {
        "records": "Showing full dataset records table.",
        "schema": "Drill-Down Active: Exploring column attributes, data types, and non-null counts.",
        "numerical": f"Drill-Down Active: Highlighting {len(meta['numerical_cols'])} continuous numerical columns.",
        "categorical": f"Drill-Down Active: Highlighting {len(meta['categorical_cols'])} discrete categorical columns.",
        "missing": f"Drill-Down Active: Filtered to rows containing missing values ({meta['missing_values']} total missing cells).",
        "duplicates": f"Drill-Down Active: Filtered to duplicate records ({meta['duplicate_rows']} duplicate rows detected)."
    }

    if target in target_descriptions and target != "all":
        st.info(f"🎯 **{target_descriptions[target]}**")

    # Quick Switcher Ribbon for instant tab view jumping
    st.markdown("##### 🧭 Explorer Views")
    q_btn1, q_btn2, q_btn3, q_btn4, q_btn5, q_btn6 = st.columns(6)
    with q_btn1:
        st.button("📋 Full Dataset", key="q_view_all", on_click=set_nav, args=("🔍 Data Explorer", "records"), use_container_width=True)
    with q_btn2:
        st.button("🏛️ Schema & Types", key="q_view_schema", on_click=set_nav, args=("🔍 Data Explorer", "schema"), use_container_width=True)
    with q_btn3:
        st.button("🔢 Numerical View", key="q_view_num", on_click=set_nav, args=("🔍 Data Explorer", "numerical"), use_container_width=True)
    with q_btn4:
        st.button("🏷️ Categorical View", key="q_view_cat", on_click=set_nav, args=("🔍 Data Explorer", "categorical"), use_container_width=True)
    with q_btn5:
        st.button("🩹 Missing Records", key="q_view_missing", on_click=set_nav, args=("🔍 Data Explorer", "missing"), use_container_width=True)
    with q_btn6:
        st.button("👥 Duplicate Rows", key="q_view_dups", on_click=set_nav, args=("🔍 Data Explorer", "duplicates"), use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ---------------- VIEW RENDERING BASED ON TARGET / TABS ----------------
    if target == "schema":
        st.markdown("### 🏛️ Column Schema, Data Types & Null Auditing")
        schema_df = pd.DataFrame({
            "Column Attribute": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notna().sum(),
            "Null Count": df.isna().sum(),
            "Null %": (df.isna().sum() / len(df) * 100).round(2).astype(str) + "%",
            "Unique Count": df.nunique(),
            "Sample Value": [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "None" for col in df.columns]
        }).reset_index(drop=True)
        st.dataframe(schema_df, use_container_width=True)
        st.caption(f"Estimated Deep Memory Allocation: **{meta['memory_usage']}** across {len(df.columns)} attributes.")

    elif target == "numerical":
        st.markdown("### 🔢 Numerical Columns Exploration")
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if not num_cols:
            st.info("No continuous numerical variables in this dataset.")
        else:
            st.markdown(f"**{len(num_cols)} Continuous Numerical Columns Detected:** `{', '.join(num_cols)}`")
            num_summary = StatisticalAnalyzer.compute_numerical_summary(df)
            st.dataframe(num_summary.T, use_container_width=True)
            
            st.markdown("#### 🔍 Filtered Numerical Records Table")
            st.dataframe(df[num_cols], use_container_width=True)

    elif target == "categorical":
        st.markdown("### 🏷️ Categorical Columns Exploration")
        cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        if not cat_cols:
            st.info("No categorical variables detected in this dataset.")
        else:
            st.markdown(f"**{len(cat_cols)} Categorical Features Detected:** `{', '.join(cat_cols)}`")
            cat_summary = StatisticalAnalyzer.compute_categorical_summary(df)
            st.dataframe(cat_summary, use_container_width=True)
            
            st.markdown("#### 🔍 Categorical Table Preview")
            st.dataframe(df[cat_cols], use_container_width=True)

    elif target == "missing":
        st.markdown("### 🩹 Rows Containing Missing / Null Values")
        rows_with_na = df[df.isna().any(axis=1)]
        if rows_with_na.empty:
            st.success("🎉 Perfect Completeness! There are zero rows with missing values in the current dataset.")
        else:
            st.markdown(f"Found **{len(rows_with_na):,} rows** containing at least one missing cell ({len(rows_with_na) / len(df) * 100:.2f}% of dataset).")
            
            c_mv_act1, c_mv_act2 = st.columns([2, 1])
            with c_mv_act1:
                st.dataframe(rows_with_na, use_container_width=True)
            with c_mv_act2:
                st.markdown("#### Missing Columns Breakdown")
                missing_counts = df.isna().sum()
                missing_counts = missing_counts[missing_counts > 0].reset_index()
                missing_counts.columns = ["Column", "Null Count"]
                missing_counts["Null %"] = (missing_counts["Null Count"] / len(df) * 100).round(2).astype(str) + "%"
                st.dataframe(missing_counts, use_container_width=True)
                
                st.button("🧹 Jump to Data Cleaning for Imputation ➔", key="btn_goto_clean_from_missing", on_click=set_nav, args=("🧹 Data Cleaning", "all"), type="primary", use_container_width=True)

    elif target == "duplicates":
        st.markdown("### 👥 Duplicate Records View")
        dup_rows = df[df.duplicated(keep=False)]
        if dup_rows.empty:
            st.success("🎉 All records are unique! Zero duplicate rows found.")
        else:
            dup_cnt = int(df.duplicated().sum())
            st.markdown(f"Found **{dup_cnt:,} duplicate rows** ({dup_cnt / len(df) * 100:.2f}% duplicate records). Showing all matching rows below:")
            st.dataframe(dup_rows, use_container_width=True)
            
            st.button("🚀 Jump to Data Cleaning to Remove Duplicates ➔", key="btn_goto_clean_from_dups", on_click=set_nav, args=("🧹 Data Cleaning", "all"), type="primary")

    else:
        # FULL DATASET WORKSPACE (Default)
        st.markdown("### 📋 Searchable Dataset Table & Advanced Slicing")
        
        # View Controls Ribbon
        c_view1, c_view2, c_view3 = st.columns([1.5, 2, 2.5])
        with c_view1:
            slice_mode = st.selectbox("Row Slicing Mode", ["Entire Dataset", "First N rows", "Last N rows", "Random Sample"], key="exp_slice_mode")
        with c_view2:
            if slice_mode in ["First N rows", "Last N rows", "Random Sample"]:
                n_rows_slice = st.slider("Row Count", min_value=5, max_value=min(len(df), 500), value=min(len(df), 50), step=5, key="exp_slice_slider")
            else:
                n_rows_slice = len(df)
        with c_view3:
            # Column Quick Select Actions
            c_qs1, c_qs2, c_qs3 = st.columns(3)
            with c_qs1:
                if st.button("Select All", key="btn_sel_all", use_container_width=True):
                    st.session_state["exp_cols"] = list(df.columns)
            with c_qs2:
                if st.button("Num Only", key="btn_sel_num", use_container_width=True):
                    st.session_state["exp_cols"] = list(df.select_dtypes(include=[np.number]).columns)
            with c_qs3:
                if st.button("Cat Only", key="btn_sel_cat", use_container_width=True):
                    st.session_state["exp_cols"] = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

            default_selected = st.session_state.get("exp_cols", list(df.columns))
            valid_selected = [c for c in default_selected if c in df.columns]
            if not valid_selected:
                valid_selected = list(df.columns)
            selected_cols = st.multiselect("Visible Columns", options=list(df.columns), default=valid_selected, key="exp_cols_select")

        # Multi-Condition Filtering Expander
        with st.expander("🛠️ Dynamic Multi-Condition Filtering (Categorical, Numerical & Global Search)", expanded=False):
            f_cols = st.columns(3)
            filtered_df = df.copy()

            # Categorical Filter
            cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
            if cat_cols:
                with f_cols[0]:
                    filter_cat_col = st.selectbox("Filter by Category", ["None"] + cat_cols, key="exp_filter_cat")
                    if filter_cat_col != "None":
                        unique_vals = list(df[filter_cat_col].dropna().unique())
                        selected_vals = st.multiselect(f"Classes in {filter_cat_col}", options=unique_vals, default=unique_vals[:min(3, len(unique_vals))], key="exp_filter_cat_vals")
                        if selected_vals:
                            filtered_df = filtered_df[filtered_df[filter_cat_col].isin(selected_vals)]

            # Numerical Range Filter
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            if num_cols:
                with f_cols[1]:
                    filter_num_col = st.selectbox("Filter by Numerical Range", ["None"] + num_cols, key="exp_filter_num")
                    if filter_num_col != "None":
                        min_val = float(df[filter_num_col].min())
                        max_val = float(df[filter_num_col].max())
                        if min_val < max_val:
                            num_range = st.slider(f"Range for {filter_num_col}", min_value=min_val, max_value=max_val, value=(min_val, max_val), key="exp_filter_num_range")
                            filtered_df = filtered_df[(filtered_df[filter_num_col] >= num_range[0]) & (filtered_df[filter_num_col] <= num_range[1])]

            # Global Search Filter
            with f_cols[2]:
                search_query = st.text_input("Global Full-Text Search", "", placeholder="e.g. Laptop, Consumer, etc.", key="exp_global_search")
                if search_query:
                    mask = np.column_stack([filtered_df[col].astype(str).str.contains(search_query, case=False, na=False) for col in filtered_df.columns])
                    filtered_df = filtered_df[mask.any(axis=1)]

        if not selected_cols:
            st.warning("Please select at least one visible column.")
        else:
            display_df = filtered_df[selected_cols]
            if slice_mode == "First N rows":
                display_df = display_df.head(n_rows_slice)
            elif slice_mode == "Last N rows":
                display_df = display_df.tail(n_rows_slice)
            elif slice_mode == "Random Sample":
                display_df = display_df.sample(min(n_rows_slice, len(display_df)))

            c_info1, c_info2 = st.columns([3, 1])
            with c_info1:
                st.markdown(f"**Showing {len(display_df):,} records** (Filtered from {len(df):,} total rows • {len(selected_cols)} active columns)")
            with c_info2:
                filtered_csv = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Filtered CSV",
                    data=filtered_csv,
                    file_name="filtered_dataset_view.csv",
                    mime="text/csv",
                    use_container_width=True
                )

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
        if st.button("↩️ Undo Last Cleaning Step", key="btn_undo_clean", disabled=len(st.session_state.df_history) == 0, use_container_width=True):
            st.session_state.df = st.session_state.df_history.pop()
            st.session_state.cleaning_history.append("Undid last operation.")
            st.toast("Reverted last cleaning operation!")
            st.rerun()
    with col_u2:
        if st.button("🔄 Reset to Original Raw Data", key="btn_reset_clean", use_container_width=True):
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
            if st.button("🚀 Eliminate All Duplicate Rows", key="btn_clean_dedup", type="primary"):
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
                    format_func=lambda x: x[1],
                    key="clean_impute_strategy"
                )[0]
                
            with c_mv2:
                target_cols = st.multiselect("Select Target Columns", options=cols_with_na, default=cols_with_na, key="clean_impute_cols")

            custom_val = None
            if strategy == "fill_custom":
                custom_val = st.text_input("Enter Custom Imputation Value", "Unknown", key="clean_impute_custom_val")

            if st.button("⚡ Execute Imputation Strategy", key="btn_clean_impute_exec", type="primary"):
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
            style = st.selectbox("Naming Convention", ["snake_case", "lowercase", "uppercase", "strip_spaces"], key="clean_header_style")
            if st.button("✨ Apply Convention to All Headers", key="btn_clean_headers"):
                prev_df = df.copy()
                cleaned_df, mapping = DataCleaner.standardize_column_names(df, style=style)
                st.session_state.df = cleaned_df
                push_history(prev_df, f"Standardized headers to {style}.")
                st.success(f"✅ Column names transformed to {style}!")
                st.rerun()

        with c_rn2:
            st.markdown("#### Manual Column Rename")
            old_col = st.selectbox("Target Column", options=list(df.columns), key="clean_rename_old")
            new_col = st.text_input("New Name", value=old_col, key="clean_rename_new")
            if st.button("✏️ Rename Column", key="btn_clean_rename_exec"):
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
            col_to_cast = st.selectbox("Select Column to Cast", options=list(df.columns), key="clean_cast_col")
        with c_tc2:
            target_type = st.selectbox("Target Type", ["integer", "float", "string", "datetime", "category", "boolean"], key="clean_cast_type")
        with c_tc3:
            st.write("")
            st.write("")
            if st.button("🔀 Execute Cast", key="btn_clean_cast_exec", type="primary"):
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
            selected_np_col = st.selectbox("Select Continuous Feature for NumPy Analysis", options=num_cols, key="stat_np_col")
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
            ],
            key="vis_chart_type"
        )
        palette = st.selectbox("Aesthetic Palette", list(Visualizer.COLOR_PALETTES.keys()), index=0, key="vis_palette")

        # Dynamic Controls per Chart Type
        x_col, y_col, hue_col = None, None, None
        bins, kde, show_trend = 20, True, False
        agg_func = "count"
        is_donut = True
        max_slices = 7

        if chart_type == "Histogram & KDE":
            if not num_cols:
                st.error("No continuous numerical variables for Histogram.")
            else:
                x_col = st.selectbox("Numerical Feature (X)", options=num_cols, key="vis_hist_x")
                bins = st.slider("Bin Granularity", 5, 50, 20, key="vis_hist_bins")
                kde = st.checkbox("Overlay KDE Density Curve", value=True, key="vis_hist_kde")

        elif chart_type == "Bar Chart":
            x_col = st.selectbox("Grouping Feature (X)", options=all_cols, key="vis_bar_x")
            agg_type = st.selectbox("Aggregation Metric", ["Frequency Count", "Sum of Y", "Mean of Y"], key="vis_bar_agg")
            if agg_type != "Frequency Count":
                y_col = st.selectbox("Numerical Metric (Y)", options=num_cols, key="vis_bar_y")
                agg_func = "sum" if "Sum" in agg_type else "mean"
            else:
                y_col = None
                agg_func = "count"

        elif chart_type == "Line Chart":
            x_col = st.selectbox("X-Axis (Time / Sequence)", options=all_cols, key="vis_line_x")
            y_col = st.selectbox("Y-Axis (Numerical Metric)", options=num_cols if num_cols else all_cols, key="vis_line_y")
            hue_opt = ["None"] + cat_cols
            hue_sel = st.selectbox("Hue Grouping (Optional)", options=hue_opt, key="vis_line_hue")
            hue_col = None if hue_sel == "None" else hue_sel

        elif chart_type == "Scatter Plot":
            if len(num_cols) < 2:
                st.error("Scatter plots require at least 2 numerical features.")
            else:
                x_col = st.selectbox("Independent Variable (X)", options=num_cols, index=0, key="vis_scat_x")
                y_col = st.selectbox("Dependent Variable (Y)", options=num_cols, index=min(1, len(num_cols)-1), key="vis_scat_y")
                hue_opt = ["None"] + cat_cols
                hue_sel = st.selectbox("Color Hue (Optional)", options=hue_opt, key="vis_scat_hue")
                hue_col = None if hue_sel == "None" else hue_sel
                show_trend = st.checkbox("Include Linear Regression Trendline", value=False, key="vis_scat_trend")

        elif chart_type == "Box & Whisker Plot":
            if not num_cols:
                st.error("Box plot requires numerical variables.")
            else:
                y_col = st.selectbox("Numerical Feature (Y)", options=num_cols, key="vis_box_y")
                x_opt = ["None"] + cat_cols
                x_sel = st.selectbox("Group by Category (X - Optional)", options=x_opt, key="vis_box_x")
                x_col = None if x_sel == "None" else x_sel

        elif chart_type == "Pie / Donut Chart":
            if not cat_cols:
                st.error("Pie charts require categorical features.")
            else:
                x_col = st.selectbox("Categorical Feature", options=cat_cols, key="vis_pie_col")
                is_donut = st.checkbox("Donut Hole Style", value=True, key="vis_pie_donut")
                max_slices = st.slider("Max Slices", 3, 10, 6, key="vis_pie_slices")

        elif chart_type == "Seaborn Count Plot":
            x_col = st.selectbox("Categorical Feature", options=cat_cols if cat_cols else all_cols, key="vis_count_col")
            hue_opt = ["None"] + cat_cols
            hue_sel = st.selectbox("Group by Hue (Optional)", options=hue_opt, key="vis_count_hue")
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
                    type="primary",
                    key="vis_download_btn"
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
                color = "#16A34A" if val > 0 else "#DC2626"
                st.markdown(f"""
                <div class="insight-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{pair['Feature 1']} ↔ {pair['Feature 2']}</strong>
                        <span style="font-weight: 800; color: {color};">r = {val:.3f}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #5B6475; margin-top: 4px;">{pair['Strength']}</div>
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
                <div style="font-weight: 700; color: #172033; font-size: 0.95rem;">{title}</div>
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
        <div class="glass-panel" style="margin-bottom: 12px;">
            <div class="kpi-label">1. Cleaned Dataset (CSV)</div>
            <p style="font-size: 0.85rem; color: #5B6475; margin-top: 6px;">
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
            key="btn_dl_cleaned_csv",
            use_container_width=True
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-panel" style="margin-bottom: 12px;">
            <div class="kpi-label">2. Statistical Summary Table</div>
            <p style="font-size: 0.85rem; color: #5B6475; margin-top: 6px;">
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
                key="btn_dl_stat_csv",
                use_container_width=True
            )
        else:
            st.info("No numerical summary available.")

    with exp_col2:
        st.markdown("""
        <div class="glass-panel" style="margin-bottom: 12px;">
            <div class="kpi-label">3. Correlation Matrix (CSV)</div>
            <p style="font-size: 0.85rem; color: #5B6475; margin-top: 6px;">
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
                key="btn_dl_corr_csv",
                use_container_width=True
            )
        else:
            st.info("No correlation matrix available.")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-panel" style="margin-bottom: 12px;">
            <div class="kpi-label">4. Comprehensive Analysis Report (.txt)</div>
            <p style="font-size: 0.85rem; color: #5B6475; margin-top: 6px;">
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
            key="btn_dl_report_txt",
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
        <div class="glass-panel">
            <div style="font-size: 1.1rem; font-weight: 800; color: #172033;">🐼 Pandas</div>
            <div style="font-size: 0.85rem; color: #5B6475; margin-top: 6px;">
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
        <div class="glass-panel">
            <div style="font-size: 1.1rem; font-weight: 800; color: #172033;">🔢 NumPy</div>
            <div style="font-size: 0.85rem; color: #5B6475; margin-top: 6px;">
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
        <div class="glass-panel">
            <div style="font-size: 1.1rem; font-weight: 800; color: #172033;">📊 Matplotlib</div>
            <div style="font-size: 0.85rem; color: #5B6475; margin-top: 6px;">
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
        <div class="glass-panel">
            <div style="font-size: 1.1rem; font-weight: 800; color: #172033;">🎨 Seaborn</div>
            <div style="font-size: 0.85rem; color: #5B6475; margin-top: 6px;">
                • Distribution KDE plots<br/>
                • Curated color palettes<br/>
                • Correlation heatmaps<br/>
                • Categorical count plots<br/>
                • Regression trendlines
            </div>
        </div>
        """, unsafe_allow_html=True)

    