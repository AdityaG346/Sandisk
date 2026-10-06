"""
dashboard/app.py
================
Interactive Streamlit Dashboard for Semiconductor Die Yield Prediction.
Presents precomputed model results, wafer maps, continuous risk fields,
per-die SHAP attributions, sub-die block readings, and Model A vs B comparisons.

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
os.environ["PYTHONUTF8"] = "1"
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.triage import topk_stats, capture_curve

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import shap
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration & SanDisk-Inspired Design System (design.md)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DieYield Intelligence | Multi-Resolution Yield Prediction",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SanDisk Industrial Design System CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --sd-red: #F51B0B;
        --sd-black: #050505;
        --sd-surface-1: #0D0F12;
        --sd-surface-2: #15181D;
        --sd-surface-3: #1C2128;
        --sd-border: #30353D;

        --sd-white: #FFFFFF;
        --sd-text-secondary: #B7BDC7;
        --sd-text-muted: #7F8792;

        --sd-success: #35D07F;
        --sd-warning: #F4B740;
        --sd-failure: #FF5A5F;

        --sd-model-a: #25B9E6;
        --sd-model-b: #A855F7;
        --sd-context: #43D17C;

        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
    }

    /* Base App Typography and Dark Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: var(--sd-black) !important;
        color: var(--sd-white) !important;
    }

    .stApp {
        background-color: var(--sd-black) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--sd-surface-1) !important;
        border-right: 1px solid var(--sd-border) !important;
    }

    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Product Header */
    .sd-header-wrap {
        background: linear-gradient(180deg, #101217 0%, var(--sd-black) 100%);
        border: 1px solid var(--sd-border);
        border-top: 3px solid var(--sd-red);
        border-radius: var(--radius-md);
        padding: 24px 28px 20px 28px;
        margin-bottom: 24px;
        position: relative;
    }

    .sd-brand-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 8px;
    }

    .sd-brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--sd-white);
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .sd-pixel-mark {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: var(--sd-red);
        margin-right: 4px;
        border-radius: 1px;
    }

    .sd-brand-subtitle {
        font-size: 0.95rem;
        color: var(--sd-text-secondary);
        margin-top: 4px;
        font-weight: 500;
    }

    .sd-meta-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid rgba(48, 53, 61, 0.6);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--sd-text-secondary);
    }

    .sd-meta-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .sd-meta-item span.val {
        color: var(--sd-white);
        font-family: 'JetBrains Mono', monospace;
    }

    /* Badges */
    .sd-badge-verified {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(53, 208, 127, 0.12);
        color: var(--sd-success);
        border: 1px solid rgba(53, 208, 127, 0.35);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 999px;
    }

    .sd-badge-verified::before {
        content: "●";
        font-size: 0.7rem;
    }

    .sd-badge-model-a {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: rgba(37, 185, 230, 0.12);
        color: var(--sd-model-a);
        border: 1px solid rgba(37, 185, 230, 0.3);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: var(--radius-sm);
    }

    .sd-badge-model-b {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: rgba(168, 85, 247, 0.12);
        color: var(--sd-model-b);
        border: 1px solid rgba(168, 85, 247, 0.35);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: var(--radius-sm);
    }

    .sd-badge-context {
        display: inline-flex;
        align-items: center;
        background-color: rgba(67, 209, 124, 0.1);
        color: var(--sd-context);
        border: 1px solid rgba(67, 209, 124, 0.25);
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: var(--radius-sm);
    }

    /* Cards */
    .sd-card {
        background-color: var(--sd-surface-2);
        border: 1px solid var(--sd-border);
        border-radius: var(--radius-md);
        padding: 16px 18px;
        transition: transform 160ms cubic-bezier(0.16, 1, 0.3, 1), border-color 160ms ease;
    }

    .sd-card:hover {
        border-color: #424954;
        transform: translateY(-1px);
    }

    /* Executive Results KPI Cards - Equal Height and Alignment */
    /* Use min-height on the row instead of 100% height on inner wrappers
       to avoid creating stacking contexts that fight the Streamlit dropdown portal */
    div[data-testid="stHorizontalBlock"]:has(.sd-exec-card) {
        align-items: stretch !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.sd-exec-card) [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.sd-exec-card) [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.sd-exec-card) [data-testid="stMarkdown"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.sd-exec-card) [data-testid="stMarkdownContainer"] {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
    }

    /* Selectbox dropdown portal — must NOT override position (JS sets it via Floating UI).
       Only set z-index so the portal floats above all page content. */
    div[data-baseweb="popover"] {
        z-index: 9999 !important;
    }

    /* Constrain the dropdown list height so it never bleeds past the viewport.
       All options remain accessible via scrolling inside the list. */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        max-height: min(320px, 40vh) !important;
        overflow-y: auto !important;
        scrollbar-width: thin !important;
        scrollbar-color: #30353D #0D0F12 !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"]::-webkit-scrollbar {
        width: 5px;
    }

    ul[data-testid="stSelectboxVirtualDropdown"]::-webkit-scrollbar-track {
        background: #0D0F12;
        border-radius: 3px;
    }

    ul[data-testid="stSelectboxVirtualDropdown"]::-webkit-scrollbar-thumb {
        background: #30353D;
        border-radius: 3px;
    }

    ul[data-testid="stSelectboxVirtualDropdown"]::-webkit-scrollbar-thumb:hover {
        background: #424954;
    }

    /* Ensure Streamlit column containers don't clip the portal */
    section[data-testid="stSidebar"] ~ div [data-testid="stVerticalBlock"],
    [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] {
        overflow: visible !important;
    }

    .sd-exec-card {
        background-color: var(--sd-surface-2);
        border: 1px solid var(--sd-border);
        border-top: 3px solid transparent;
        border-radius: var(--radius-md);
        padding: 12px 14px 10px 14px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        min-height: 136px;
        width: 100%;
        transition: transform 160ms cubic-bezier(0.16, 1, 0.3, 1), border-color 160ms ease;
    }

    .sd-exec-card:hover {
        border-color: #424954;
        transform: translateY(-1px);
    }

    /* 10% Inspection Budget (Hero) - Highlighted, same height */
    .sd-exec-card-hero {
        background: linear-gradient(180deg, #1a1e26 0%, var(--sd-surface-2) 100%);
        border: 1px solid var(--sd-border);
        border-top: 3px solid var(--sd-red);
        box-shadow: 0 4px 20px rgba(245, 27, 11, 0.08);
    }

    .sd-exec-card-hero:hover {
        border-color: #4d5563;
        border-top-color: var(--sd-red);
        transform: translateY(-1px);
    }

    .sd-card-hero {
        background: linear-gradient(180deg, #1a1e26 0%, var(--sd-surface-2) 100%);
        border: 1px solid var(--sd-border);
        border-top: 3px solid var(--sd-red);
        border-radius: var(--radius-md);
        padding: 16px 18px;
        box-shadow: 0 4px 20px rgba(245, 27, 11, 0.08);
        transition: transform 160ms cubic-bezier(0.16, 1, 0.3, 1), border-color 160ms ease;
    }

    .sd-card-hero:hover {
        border-color: #4d5563;
        transform: translateY(-1px);
    }

    .sd-kpi-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--sd-text-secondary);
        margin-bottom: 2px;
        line-height: 1.2;
    }

    .sd-kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--sd-white);
        line-height: 1.15;
        margin-bottom: 2px;
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: baseline;
        gap: 4px;
    }

    .sd-kpi-value-hero {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--sd-white);
        line-height: 1.15;
        margin-bottom: 2px;
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: baseline;
        gap: 4px;
    }

    .sd-kpi-sub-val {
        font-size: 0.78rem;
        color: var(--sd-text-secondary);
        font-weight: 500;
    }

    .sd-kpi-delta {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--sd-success);
        display: flex;
        align-items: center;
        gap: 3px;
        line-height: 1.2;
    }

    .sd-kpi-delta-red {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--sd-red);
        line-height: 1.2;
    }

    .sd-kpi-sub {
        font-size: 0.65rem;
        color: var(--sd-text-muted);
        margin-top: 2px;
        line-height: 1.22;
    }

    /* Section Headers */
    .sd-section-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--sd-white);
        margin-top: 10px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sd-section-sub {
        font-size: 0.88rem;
        color: var(--sd-text-secondary);
        margin-bottom: 16px;
        line-height: 1.45;
    }

    .sd-divider {
        height: 1px;
        background-color: var(--sd-border);
        margin: 32px 0 24px 0;
        position: relative;
    }

    .sd-divider::before {
        content: "";
        position: absolute;
        left: 0;
        top: -1px;
        width: 8px;
        height: 3px;
        background-color: var(--sd-red);
    }

    /* Architecture Flow Box */
    .sd-arch-container {
        background-color: var(--sd-surface-2);
        border: 1px solid var(--sd-border);
        border-radius: var(--radius-md);
        padding: 22px 24px;
        margin-bottom: 20px;
    }

    /* Limitations & Disclosures */
    .sd-disclaimer {
        background-color: var(--sd-surface-1);
        border: 1px solid var(--sd-border);
        border-left: 3px solid var(--sd-warning);
        padding: 14px 18px;
        border-radius: var(--radius-sm);
        font-size: 0.82rem;
        color: var(--sd-text-secondary);
        line-height: 1.5;
        margin: 14px 0;
    }

    .sd-limitations {
        background-color: var(--sd-surface-2);
        border: 1px solid var(--sd-border);
        border-left: 3px solid var(--sd-model-b);
        padding: 18px 22px;
        border-radius: var(--radius-md);
        font-size: 0.85rem;
        color: var(--sd-text-secondary);
        line-height: 1.6;
        margin: 20px 0;
    }

    /* Head-to-Head Comparison Card */
    .sd-hero-die-card {
        background: linear-gradient(180deg, #181c22 0%, var(--sd-surface-2) 100%);
        border: 1px solid var(--sd-border);
        border-top: 2px solid var(--sd-model-b);
        border-radius: var(--radius-md);
        padding: 20px 22px;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--sd-border) !important;
        border-radius: var(--radius-md) !important;
        overflow: auto !important;
    }

    /* Responsive Comparison Tables Grid */
    .sd-table-comparison-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
        gap: 18px;
        width: 100%;
        margin-top: 14px;
        margin-bottom: 24px;
        box-sizing: border-box;
        align-items: start;
    }

    @media (max-width: 1100px) {
        .sd-table-comparison-grid {
            grid-template-columns: 1fr !important;
            gap: 18px;
        }
    }

    .sd-perf-card {
        background-color: var(--sd-surface-2);
        border: 1px solid var(--sd-border);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        box-sizing: border-box;
        width: 100%;
        display: flex;
        flex-direction: column;
        transition: border-color 160ms ease;
    }

    .sd-perf-card:hover {
        border-color: #424954;
    }

    .sd-perf-card-header {
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(48, 53, 61, 0.6);
    }

    .sd-perf-card-topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }

    .sd-perf-card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--sd-white);
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sd-perf-card-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--sd-text-muted);
        background-color: var(--sd-surface-1);
        border: 1px solid rgba(48, 53, 61, 0.6);
        padding: 2px 7px;
        border-radius: var(--radius-sm);
    }

    .sd-perf-card-subtitle {
        font-size: 0.78rem;
        color: var(--sd-text-secondary);
        line-height: 1.35;
    }

    .sd-perf-table-wrapper {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border: 1px solid var(--sd-border);
        border-radius: var(--radius-sm);
        background-color: var(--sd-surface-1);
    }

    .sd-perf-table-wrapper::-webkit-scrollbar {
        height: 6px;
        width: 6px;
    }

    .sd-perf-table-wrapper::-webkit-scrollbar-track {
        background: var(--sd-surface-1);
    }

    .sd-perf-table-wrapper::-webkit-scrollbar-thumb {
        background: var(--sd-border);
        border-radius: 3px;
    }

    .sd-perf-table-wrapper::-webkit-scrollbar-thumb:hover {
        background: #424954;
    }

    .sd-perf-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
        text-align: left;
    }

    .sd-perf-table th {
        background-color: var(--sd-surface-2);
        color: var(--sd-text-secondary);
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 7px 6px;
        border-bottom: 1px solid var(--sd-border);
        line-height: 1.25;
        vertical-align: bottom;
        white-space: normal;
    }

    .sd-perf-table th.sd-th-num {
        text-align: right;
    }

    .sd-th-sub {
        font-size: 0.60rem;
        font-weight: 500;
        color: var(--sd-text-muted);
        text-transform: none;
        letter-spacing: normal;
        display: block;
        margin-top: 1px;
    }

    .sd-perf-table td {
        padding: 7px 6px;
        border-bottom: 1px solid rgba(48, 53, 61, 0.45);
        color: var(--sd-text-primary);
        font-size: 0.78rem;
        line-height: 1.25;
        vertical-align: middle;
    }

    .sd-perf-table td.sd-td-num {
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }

    .sd-perf-table tr:last-child td {
        border-bottom: none;
    }

    .sd-perf-table tr:hover td {
        background-color: rgba(28, 33, 40, 0.55);
    }

    .sd-perf-table tr.sd-row-highlight td {
        background-color: rgba(168, 85, 247, 0.06);
    }

    .sd-perf-table tr.sd-row-highlight:hover td {
        background-color: rgba(168, 85, 247, 0.12);
    }

    .sd-td-sub {
        font-size: 0.65rem;
        color: var(--sd-text-secondary);
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
        margin-top: 1px;
    }

    .sd-td-sub-pos {
        font-size: 0.65rem;
        color: var(--sd-success);
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
        margin-top: 1px;
    }

    .sd-td-sub-neg {
        font-size: 0.65rem;
        color: #F87171;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
        margin-top: 1px;
    }

    .sd-td-sub-muted {
        font-size: 0.65rem;
        color: var(--sd-text-muted);
        line-height: 1.1;
        margin-top: 1px;
    }

    .sd-model-sub {
        font-size: 0.68rem;
        color: var(--sd-text-secondary);
        margin-top: 2px;
        line-height: 1.15;
    }

    .sd-perf-card-footer {
        margin-top: 10px;
        font-size: 0.72rem;
        color: var(--sd-text-muted);
        line-height: 1.35;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .sd-metric-pos {
        color: var(--sd-success);
        font-weight: 700;
        font-size: 0.72rem;
        margin-left: 2px;
    }

    .sd-metric-note {
        color: var(--sd-text-muted);
        font-size: 0.70rem;
        margin-left: 2px;
    }

    /* Streamlit Widget Overrides */
    div.stSelectbox > div > div {
        background-color: var(--sd-surface-2) !important;
        border: 1px solid var(--sd-border) !important;
        color: var(--sd-white) !important;
        border-radius: var(--radius-sm) !important;
    }

    div.stSelectbox > div > div:focus-within {
        border-color: var(--sd-red) !important;
        box-shadow: 0 0 0 2px rgba(245, 27, 11, 0.2) !important;
    }

    div.stRadio > div {
        background-color: transparent !important;
    }

    /* Prevent text truncation in any Streamlit native metrics */
    div[data-testid="stMetric"] {
        background-color: transparent !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--sd-text-secondary) !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: var(--sd-white) !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        white-space: normal !important;
        overflow: visible !important;
    }

    /* Radio Label Wrap Override */
    div.stRadio div[role="radiogroup"] label {
        white-space: normal !important;
    }

    /* Triage Comparison Cards */
    .sd-triage-card {
        background-color: var(--sd-surface-2);
        border: 1px solid var(--sd-border);
        border-radius: var(--radius-md);
        padding: 18px 20px;
        margin-bottom: 20px;
        transition: border-color 160ms ease, transform 160ms cubic-bezier(0.16, 1, 0.3, 1);
    }

    .sd-triage-card:hover {
        border-color: #424954;
        transform: translateY(-1px);
    }

    .sd-triage-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(48, 53, 61, 0.6);
    }

    .sd-triage-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--sd-white);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sd-triage-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px 14px;
    }

    .sd-triage-cell {
        background-color: var(--sd-surface-1);
        border: 1px solid rgba(48, 53, 61, 0.6);
        border-radius: var(--radius-sm);
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 72px;
    }

    .sd-triage-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--sd-text-secondary);
        margin-bottom: 4px;
        line-height: 1.25;
    }

    .sd-triage-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--sd-white);
        line-height: 1.2;
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 4px;
    }

    .sd-triage-val-compact {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--sd-white);
        line-height: 1.2;
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 4px;
    }

    .sd-triage-sub {
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--sd-text-secondary);
    }

    .sd-triage-delta {
        font-size: 0.74rem;
        font-weight: 600;
        margin-top: 4px;
        display: inline-flex;
        align-items: center;
        gap: 3px;
        line-height: 1.2;
    }

    .sd-triage-delta-pos {
        color: var(--sd-success);
    }

    .sd-triage-delta-neutral {
        color: var(--sd-text-muted);
    }

    /* Product Footer */
    .sd-footer {
        margin-top: 48px;
        padding-top: 16px;
        border-top: 1px solid var(--sd-border);
        font-size: 0.75rem;
        color: var(--sd-text-muted);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)


REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
INPUT_DIR = REPO_ROOT / "input"


# -----------------------------------------------------------------------------
# Cached Artifact Loaders (Zero-Retraining, Memory-Optimized)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading test dataset metadata...")
def load_test_metadata() -> pd.DataFrame:
    """Load core test metadata (wafer_id, die_row, die_col, old_label, label)."""
    meta_pq = CACHE_DIR / "test_meta.parquet"
    if meta_pq.exists():
        return pd.read_parquet(meta_pq)
    
    test_csv = INPUT_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test dataset not found at {test_csv} or {meta_pq}!")
    cols = ["wafer_id", "die_row", "die_col", "old_label"]
    df_sample = pd.read_csv(test_csv, nrows=1)
    if "label" in df_sample.columns:
        cols.append("label")
    df = pd.read_csv(test_csv, usecols=cols)
    return df


@st.cache_data(show_spinner="Loading precomputed predictions...")
def load_predictions() -> pd.DataFrame:
    """Load final predictions CSV."""
    pred_path = OUTPUT_DIR / "predictions.csv"
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return pd.DataFrame()


@st.cache_resource(show_spinner="Loading production models...")
def load_models() -> Tuple[object, dict, object, dict]:
    """Load Model A and Model B with metadata."""
    model_a = joblib.load(OUTPUT_DIR / "model_a.pkl")
    meta_a = joblib.load(OUTPUT_DIR / "model_a_meta.pkl")
    model_b = joblib.load(OUTPUT_DIR / "model_b.pkl")
    meta_b = joblib.load(OUTPUT_DIR / "model_b_meta.pkl")
    return model_a, meta_a, model_b, meta_b


@st.cache_data(show_spinner="Loading precomputed probability vectors...")
def load_probabilities() -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-scored probabilities for Model A and Model B."""
    prob_a_path = CACHE_DIR / "test_probs_a.npy"
    prob_b_path = CACHE_DIR / "test_probs_b.npy"
    if prob_a_path.exists() and prob_b_path.exists():
        return np.load(prob_a_path), np.load(prob_b_path)
    
    # Fallback compute if cache files are missing
    test_df = load_test_metadata()
    model_a, meta_a, model_b, meta_b = load_models()
    X_a = pd.read_parquet(CACHE_DIR / "X_test_a.parquet")
    X_b = pd.read_parquet(CACHE_DIR / "X_test_b.parquet")
    prob_a = model_a.predict_proba(X_a[meta_a["feat_cols"]].values)[:, 1]
    prob_b = model_b.predict_proba(X_b[meta_b["feat_cols"]].values)[:, 1]
    return prob_a, prob_b


@st.cache_data(show_spinner="Loading calibrated probability vectors...")
def load_calibrated_probabilities() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load Platt calibrated probabilities if available."""
    cal_a_path = CACHE_DIR / "test_probs_a_cal.npy"
    cal_b_path = CACHE_DIR / "test_probs_b_cal.npy"
    prob_a_cal = np.load(cal_a_path) if cal_a_path.exists() else None
    prob_b_cal = np.load(cal_b_path) if cal_b_path.exists() else None
    return prob_a_cal, prob_b_cal


@st.cache_data(show_spinner="Loading benchmark evaluation tables...")
def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load comparison, multiseed ablation, and failure signatures tables."""
    comp_df = pd.read_csv(OUTPUT_DIR / "comparison_table.csv")
    abl_df = pd.read_csv(OUTPUT_DIR / "ablation_table_multiseed.csv")
    sig_path = OUTPUT_DIR / "failure_signatures.csv"
    sig_df = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    return comp_df, abl_df, sig_df


@st.cache_data(show_spinner="Loading sub-die block readings...")
def load_wafer_block_readings(wafer_id: str) -> pd.DataFrame:
    """Load block readings for a specific wafer."""
    w_blk_path = CACHE_DIR / "wafer_blocks" / f"{wafer_id}.parquet"
    if w_blk_path.exists():
        return pd.read_parquet(w_blk_path)
    test_csv = INPUT_DIR / "test.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv, usecols=["wafer_id", "die_row", "die_col", "block_readings"])
        return df[df["wafer_id"] == wafer_id]
    return pd.DataFrame()


@st.cache_data(show_spinner="Computing SHAP explanations for selected wafer...")
def compute_wafer_shap(wafer_id: str, model_type: str = "Model B") -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Compute and cache SHAP values for all dies on the selected wafer."""
    wafer_feat_path = CACHE_DIR / "wafer_features" / f"{wafer_id}.parquet"
    if not wafer_feat_path.exists():
        raise FileNotFoundError(f"Wafer features not found at {wafer_feat_path}!")

    X_wafer_full = pd.read_parquet(wafer_feat_path)

    model_a, meta_a, model_b, meta_b = load_models()
    if model_type == "Model A":
        model = model_a
        cols = meta_a["feat_cols"]
    else:
        model = model_b
        cols = meta_b["feat_cols"]

    X_wafer = X_wafer_full[cols]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_wafer.values)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    return shap_vals, cols, X_wafer


@st.cache_data(show_spinner="Loading population medians...")
def load_pass_medians() -> Tuple[pd.Series, pd.Series]:
    """Load feature medians of passing dies for counterfactual evaluation."""
    med_b = pd.read_parquet(CACHE_DIR / "pass_medians_b.parquet")["median"]
    med_a = pd.read_parquet(CACHE_DIR / "pass_medians_a.parquet")["median"]
    return med_a, med_b


# -----------------------------------------------------------------------------
# Precomputed Representative Benchmark Explanations (Part 4)
# -----------------------------------------------------------------------------
PRECOMPUTED_REPRESENTATIVES = {
    ("W_F_0014", 40, 18): {
        "title": "Die 1: Edge-Region Die (Wafer W_F_0014, Row 40, Col 18)",
        "probability": 0.905,
        "confidence": "Moderate Confidence (ensemble σ = 0.101)",
        "cluster": "Cluster 2 (Defect Neighborhood Proximity & Spatial Clustering)",
        "top_features": [
            ("sp_dist_to_fail", -1.0208),
            ("blk_mean", +0.7047),
            ("blk_q75", +0.3758),
        ],
        "trajectory": "Original (90.5%) → Normalizing blk_mean (81.8%) → +blk_q75 (76.2%)",
        "reduction": "14.3% risk reduction to 76.2%",
    },
    ("W_F_0019", 22, 22): {
        "title": "Die 2: Wafer-Interior Die (Wafer W_F_0019, Row 22, Col 22)",
        "probability": 0.740,
        "confidence": "Moderate Confidence (ensemble σ = 0.095)",
        "cluster": "Cluster 2 (Defect Neighborhood Proximity & Spatial Clustering)",
        "top_features": [
            ("sp_dist_to_fail", -0.9181),
            ("blk_mean", +0.2001),
            ("blk_std", -0.1873),
        ],
        "trajectory": "Original (74.0%) → Normalizing blk_mean (63.0%) → +blk_std (69.7%)",
        "reduction": "4.3% risk reduction to 69.7%",
    },
    ("W_F_0009", 9, 39): {
        "title": "Die 3: High Anomaly-Score Die (Wafer W_F_0009, Row 9, Col 39)",
        "probability": 0.883,
        "confidence": "High Confidence (ensemble σ = 0.029)",
        "cluster": "Cluster 2 (Defect Neighborhood Proximity & Spatial Clustering)",
        "top_features": [
            ("sp_dist_to_fail", -0.9906),
            ("blk_mean", +0.9579),
            ("feature_291", -0.2176),
        ],
        "trajectory": "Original (88.3%) → Normalizing blk_mean (72.3%) → +feature_291 (75.0%)",
        "reduction": "13.3% risk reduction to 75.0%",
    },
    ("W_F_0010", 48, 6): {
        "title": "Die 4: Spatial Defect-Cluster Die (Wafer W_F_0010, Row 48, Col 6)",
        "probability": 0.984,
        "confidence": "High Confidence (ensemble σ = 0.012)",
        "cluster": "Cluster -1 (Mixed / Boundary Pattern)",
        "top_features": [
            ("blk_mean", +0.7140),
            ("sp_dist_to_fail", -0.5571),
            ("feature_97", -0.1885),
        ],
        "trajectory": "Original (98.4%) → Normalizing blk_mean (97.1%) → +feature_97 (95.5%)",
        "reduction": "2.9% risk reduction to 95.5%",
    },
    ("W_F_0016", 21, 12): {
        "title": "Die 5: Marginal Threshold Case (Wafer W_F_0016, Row 21, Col 12)",
        "probability": 0.521,
        "confidence": "Moderate Confidence (ensemble σ = 0.079)",
        "cluster": "Cluster -1 (Mixed / Boundary Pattern)",
        "top_features": [
            ("blk_mean", +1.4180),
            ("sp_dist_to_fail", -1.0596),
            ("blk_q75", +0.2558),
        ],
        "trajectory": "Original (52.1%) → Normalizing blk_mean (18.3%, Crosses Boundary to Pass!) → +blk_q75 (13.5%)",
        "reduction": "38.6% risk reduction (flips to Pass at 18.3%)",
    },
}


# -----------------------------------------------------------------------------
# Visualization Rendering Functions (SanDisk Design System)
# -----------------------------------------------------------------------------
def render_wafer_4panel(
    df_wafer: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    wafer_id: str,
    model_name: str,
    has_ground_truth: bool = True,
) -> matplotlib.figure.Figure:
    """
    Render 4-panel wafer map matching design.md Section 18:
    1. PRE-TEST: What was known before burn-in
    2. GROUND TRUTH: What actually failed
    3. MODEL RISK: What the model predicts
    4. SPATIAL FIELD: Where risk clusters
    """
    rows = df_wafer["die_row"].values.astype(int)
    cols = df_wafer["die_col"].values.astype(int)
    n_r = int(rows.max()) + 1
    n_c = int(cols.max()) + 1

    pretest_grid = np.full((n_r, n_c), np.nan)
    newfail_grid = np.full((n_r, n_c), np.nan)
    prob_grid = np.full((n_r, n_c), np.nan)

    old_labels = df_wafer["old_label"].values
    has_lbl_col = has_ground_truth and ("label" in df_wafer.columns)
    labels = df_wafer["label"].values if has_lbl_col else np.zeros(len(df_wafer))

    for r, c, old_l, true_l, p in zip(rows, cols, old_labels, labels, y_prob):
        pretest_grid[r, c] = old_l
        if old_l == 0:
            if has_lbl_col:
                newfail_grid[r, c] = true_l
            prob_grid[r, c] = p

    # Continuous 2D Gaussian blur for risk-field
    prob_filled = np.nan_to_num(prob_grid, nan=0.0)
    valid_mask = ~np.isnan(prob_grid)
    norm_mask = gaussian_filter(valid_mask.astype(float), sigma=1.5)
    norm_mask = np.where(norm_mask > 0.05, norm_mask, 1.0)
    smoothed_risk = gaussian_filter(prob_filled, sigma=1.5) / norm_mask
    smoothed_risk[~valid_mask] = np.nan

    # Hotspots: regions with risk in top 10%
    valid_risks = smoothed_risk[valid_mask]
    hotspot_thresh = np.percentile(valid_risks, 90) if len(valid_risks) > 0 else 0.5
    hotspot_mask = (smoothed_risk >= hotspot_thresh) & valid_mask

    # Setup dark theme Matplotlib figure (SanDisk design tokens)
    n_panels = 4 if has_lbl_col else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6), facecolor="#15181D")
    for ax in axes:
        ax.set_facecolor("#0D0F12")
        ax.tick_params(colors="#7F8792", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30353D")

    # Panel 1: Pre-test map
    cmap_pre = mcolors.ListedColormap(["#35D07F", "#FF5A5F"])
    axes[0].imshow(pretest_grid, cmap=cmap_pre, vmin=0, vmax=1, aspect="equal")
    axes[0].set_title("1. PRE-TEST\nWhat was known before burn-in (Green=Pass, Red=Old Fail)", color="#FFFFFF", fontsize=9, pad=8, fontweight="bold")

    idx_curr = 1
    # Panel 2: Ground Truth (if available)
    if has_lbl_col:
        cmap_new = mcolors.ListedColormap(["#22272E", "#FF5A5F"])
        axes[idx_curr].imshow(newfail_grid, cmap=cmap_new, vmin=0, vmax=1, aspect="equal")
        axes[idx_curr].set_title("2. GROUND TRUTH\nWhat actually failed (Red=New Fail, Grey=Pass)", color="#FFFFFF", fontsize=9, pad=8, fontweight="bold")
        idx_curr += 1

    # Panel 3: Predicted Probability
    im_prob = axes[idx_curr].imshow(prob_grid, cmap="plasma", vmin=0, vmax=1, aspect="equal")
    axes[idx_curr].set_title(f"{idx_curr+1}. MODEL RISK ({model_name})\nPredicted failure probability (Threshold: {threshold:.3f})", color="#FFFFFF", fontsize=9, pad=8, fontweight="bold")
    cb1 = plt.colorbar(im_prob, ax=axes[idx_curr], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="#B7BDC7", labelsize=7)
    idx_curr += 1

    # Panel 4: Risk Field with Hotspot Contours
    vmax_risk = max(float(np.nanmax(smoothed_risk)) if np.any(~np.isnan(smoothed_risk)) else 0.8, 0.8)
    im_risk = axes[idx_curr].imshow(smoothed_risk, cmap="magma", vmin=0, vmax=vmax_risk, aspect="equal")
    if np.any(hotspot_mask):
        axes[idx_curr].contour(hotspot_mask, levels=[0.5], colors=["#25B9E6"], linewidths=[1.5])
    axes[idx_curr].set_title(f"{idx_curr+1}. SPATIAL FIELD\nWhere risk clusters (Cyan = Top 10% Hotspot)", color="#FFFFFF", fontsize=9, pad=8, fontweight="bold")
    cb2 = plt.colorbar(im_risk, ax=axes[idx_curr], fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#B7BDC7", labelsize=7)

    plt.tight_layout()
    return fig


def render_block_strip(
    readings: np.ndarray,
    die_row: int,
    die_col: int,
    wafer_id: str,
    mad_k: float = 2.0,
    base_mean: float = 100.0,
) -> matplotlib.figure.Figure:
    """
    Render strip plot of 2,000 sub-die block readings using robust MAD-based
    anomaly threshold (|readings - median| > 2.0 * MAD) with SanDisk styling.
    """
    n = len(readings)
    indices = np.arange(n)

    median_ = float(np.median(readings))
    mad = float(np.median(np.abs(readings - median_)))
    if mad > 0:
        anom_mask = np.abs(readings - median_) > mad_k * mad
        lo_bound = median_ - mad_k * mad
        hi_bound = median_ + mad_k * mad
    else:
        anom_mask = np.zeros(n, dtype=bool)
        lo_bound = median_
        hi_bound = median_

    fig, ax = plt.subplots(figsize=(13, 3.8), facecolor="#15181D")
    ax.set_facecolor("#0D0F12")
    ax.tick_params(colors="#7F8792", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30353D")

    ax.plot(indices, readings, color="#25B9E6", linewidth=0.55, alpha=0.9, label="Sub-die block readings")
    if anom_mask.any():
        ax.scatter(
            indices[anom_mask], readings[anom_mask],
            color="#FF5A5F", s=14, zorder=5, label=f"Anomalous readings (>{mad_k}×MAD: {anom_mask.sum():,} blocks / {100*anom_mask.mean():.1f}%)"
        )

    ax.axhline(lo_bound, color="#F4B740", linestyle="--", linewidth=0.9, label=f"Robust threshold: median ± {mad_k}×MAD [{lo_bound:.1f}, {hi_bound:.1f}]")
    ax.axhline(hi_bound, color="#F4B740", linestyle="--", linewidth=0.9)
    ax.axhline(median_, color="#35D07F", linestyle="-", linewidth=1.0, alpha=0.85, label=f"Die median ({median_:.1f})")
    ax.axhline(base_mean, color="#7F8792", linestyle=":", linewidth=0.8, alpha=0.6, label=f"Base population mean ({base_mean:.1f})")

    ax.set_xlabel("Sequential Block Array Index (0 to 1999 — NOTE: Index-Position only, NOT physical die geometry)", color="#B7BDC7", fontsize=9)
    ax.set_ylabel("Signal Reading Value", color="#B7BDC7", fontsize=9)
    ax.set_title(
        f"Die ({die_row}, {die_col}) on Wafer {wafer_id} — Sub-Die Block Signal Profile\n"
        f"Robust Anomalous Blocks (|x - median| > {mad_k}×MAD): {anom_mask.sum():,} / {n:,} ({100*anom_mask.mean():.2f}%)",
        color="#FFFFFF", fontsize=10, pad=8, fontweight="bold"
    )
    ax.legend(facecolor="#15181D", edgecolor="#30353D", labelcolor="#FFFFFF", fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    # -------------------------------------------------------------------------
    # PRODUCT HEADER (design.md Section 14)
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="sd-header-wrap">
        <div class="sd-brand-row">
            <div>
                <h1 class="sd-brand-title"><span class="sd-pixel-mark"></span>DIEYIELD INTELLIGENCE</h1>
                <div class="sd-brand-subtitle">Multi-Resolution Yield Prediction &amp; Root-Cause Analysis</div>
            </div>
            <div>
                <span class="sd-badge-verified">MODEL VERIFIED</span>
            </div>
        </div>
        <div class="sd-meta-strip">
            <div class="sd-meta-item">TEST POPULATION: <span class="val">40 HELD-OUT WAFERS</span></div>
            <div>·</div>
            <div class="sd-meta-item">ELIGIBLE DIES: <span class="val">32,598</span></div>
            <div>·</div>
            <div class="sd-meta-item">NEW FAILURES: <span class="val">1,380 (4.23%)</span></div>
            <div>·</div>
            <div class="sd-meta-item">RESOLUTION: <span class="val">MODEL A (511) → MODEL B (531)</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sd-disclaimer">
        <b>Data Grounding &amp; Architecture Scope</b>: Real WM-811K wafer geometry and pre-test maps combined with synthetic 500-channel parametric tests, 2,000-reading sub-die block telemetry, and controlled post-burn-in failure labels.
    </div>
    """, unsafe_allow_html=True)

    # 1. Load data and models
    meta_df = load_test_metadata()
    pred_df = load_predictions()
    model_a, meta_a, model_b, meta_b = load_models()
    prob_a, prob_b = load_probabilities()
    prob_a_cal, prob_b_cal = load_calibrated_probabilities()
    comp_df, abl_df, sig_df = load_tables()
    pass_med_a, pass_med_b = load_pass_medians()

    wafers = sorted(meta_df["wafer_id"].unique())

    # -------------------------------------------------------------------------
    # SECTION 1: Executive Results KPI Cards (design.md Section 16)
    # -------------------------------------------------------------------------
    st.markdown('<div class="sd-section-title">Executive Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-sub">Model B captures more failures at the same inspection budget. Adding sub-die evidence improves continuous risk ranking without increasing the inspection budget.</div>', unsafe_allow_html=True)

    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
    with col_ex1:
        st.markdown(
            '<div class="sd-exec-card">'
            '<div>'
            '<div class="sd-kpi-label">PR-AUC (Continuous Ranking)</div>'
            '<div class="sd-kpi-value">0.5362</div>'
            '</div>'
            '<div>'
            '<div class="sd-kpi-delta">Δ +0.0337 (+6.7% vs A: 0.5025)</div>'
            '<div class="sd-kpi-sub">95% wafer-cluster bootstrap CI [+0.023, +0.043]</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_ex2:
        st.markdown(
            '<div class="sd-exec-card sd-exec-card-hero">'
            '<div>'
            '<div class="sd-kpi-label" style="color: var(--sd-red);">10% Inspection Budget (Hero)</div>'
            '<div class="sd-kpi-value-hero">834 <span class="sd-kpi-sub-val">vs 768 fails</span></div>'
            '</div>'
            '<div>'
            '<div class="sd-kpi-delta-red">+66 Additional Failures Captured</div>'
            '<div class="sd-kpi-sub">60.4% vs 55.7% failure capture · 3,260 dies screened</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_ex3:
        st.markdown(
            '<div class="sd-exec-card">'
            '<div>'
            '<div class="sd-kpi-label">Test Population</div>'
            '<div class="sd-kpi-value">32,598</div>'
            '</div>'
            '<div>'
            '<div class="sd-kpi-delta" style="color: var(--sd-text-secondary);">1,380 New Failures (4.23%)</div>'
            '<div class="sd-kpi-sub">40 Held-Out Production Wafers</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_ex4:
        st.markdown(
            '<div class="sd-exec-card">'
            '<div>'
            '<div class="sd-kpi-label">Model Resolution</div>'
            '<div class="sd-kpi-value">531 <span class="sd-kpi-sub-val">Feats</span></div>'
            '</div>'
            '<div>'
            '<div class="sd-kpi-delta" style="color: var(--sd-model-b);">Model B (+20 Sub-Die Feats)</div>'
            '<div class="sd-kpi-sub">Model A: 511 features (Baseline)</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
    st.caption(":material/push_pin: **Note on Binary Decision Boundary (F1)**: Fail F1 is essentially unchanged (Model A: 0.5207 → Model B: 0.5222, Δ +0.0015; 95% bootstrap CI [−0.008, +0.012] contains zero). Model B's primary value is in continuous probability ranking and screening efficiency: at the same 10% inspection budget, Model B captures 66 more failures than Model A (834 vs. 768 failures, or 60.4% vs. 55.7%), without increasing the inspection budget.")

    # -------------------------------------------------------------------------
    # SECTION 2: Model A vs Model B Architecture & Benchmark (design.md Section 17)
    # -------------------------------------------------------------------------
    st.markdown('<div class="sd-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-title">Model Architecture: Multi-Resolution Progression</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-sub">Adding 2,000 sub-die measurements per die improves failure risk prediction beyond die-level and spatial context.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sd-arch-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div style="flex: 1; min-width: 260px; background: #0D0F12; border: 1px solid #30353D; border-left: 3px solid #25B9E6; border-radius: 6px; padding: 16px 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">MODEL A (Baseline)</span>
                    <span class="sd-badge-model-a">511 Features</span>
                </div>
                <div style="font-size: 0.82rem; color: #B7BDC7; line-height: 1.5;">
                    500 Die Parametric Electrical Tests<br>
                    + 10 Spatial Neighborhood Features<br>
                    + 1 Die-level Anomaly Detector
                </div>
            </div>
            <div style="text-align: center; padding: 0 12px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #A855F7; letter-spacing: 0.06em; margin-bottom: 4px;">+20 SUB-DIE SIGNALS</div>
                <div style="color: #A855F7; font-size: 1.5rem; font-weight: 800;">→</div>
            </div>
            <div style="flex: 1; min-width: 260px; background: #0D0F12; border: 1px solid #30353D; border-left: 3px solid #A855F7; border-radius: 6px; padding: 16px 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">MODEL B (Multi-Resolution)</span>
                    <span class="sd-badge-model-b">531 Features</span>
                </div>
                <div style="font-size: 0.82rem; color: #B7BDC7; line-height: 1.5;">
                    Model A Features (511)<br>
                    + 19 Sub-Die Block Summary Statistics (from 2,000 readings/die)<br>
                    + 1 Sub-Die Block Anomaly Detector
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Shared Responsive Table Comparison Section (Holdout + Ablation)
    # -------------------------------------------------------------------------
    f1_a = comp_df.iloc[0]['fail_f1']
    f1_b = comp_df.iloc[1]['fail_f1']
    prauc_a = comp_df.iloc[0]['pr_auc']
    prauc_b = comp_df.iloc[1]['pr_auc']
    prauc_delta = prauc_b - prauc_a
    rec_a = comp_df.iloc[0]['fail_recall'] * 100
    rec_b = comp_df.iloc[1]['fail_recall'] * 100
    rec_delta = rec_b - rec_a
    prec_a = comp_df.iloc[0]['fail_precision'] * 100
    prec_b = comp_df.iloc[1]['fail_precision'] * 100
    prec_delta = prec_b - prec_a
    th_a = meta_a['threshold']
    th_b = meta_b['threshold']

    # Build rows for 5-Seed Validation Ablation
    abl_rows = []
    for _, row in abl_df.iterrows():
        feat_name = str(row["Feature Set"])
        is_highlight = "Block" in feat_name and "Spatial" in feat_name
        row_cls = ' class="sd-row-highlight"' if is_highlight else ''
        if is_highlight:
            badge_html = ' <span class="sd-badge-model-b" style="font-size:0.60rem;padding:1px 5px;margin-left:3px;">Model B</span>'
            name_html = f'<span style="font-weight:700;color:#FFFFFF;">{feat_name}</span>{badge_html}'
            prauc_main = f'<span style="color:#43D17C;font-weight:700;">{row["PR-AUC mean"]:.4f}</span>'
        elif feat_name == "Die + Spatial":
            badge_html = ' <span class="sd-badge-model-a" style="font-size:0.60rem;padding:1px 5px;margin-left:3px;">Model A</span>'
            name_html = f'<span style="font-weight:600;color:#E1E4EA;">{feat_name}</span>{badge_html}'
            prauc_main = f'{row["PR-AUC mean"]:.4f}'
        else:
            name_html = f'<span style="color:#C9D1D9;">{feat_name}</span>'
            prauc_main = f'{row["PR-AUC mean"]:.4f}'

        prauc_sub = f'<div class="sd-td-sub">±{row["PR-AUC std"]:.4f}</div>'
        f1_main = f'{row["Fail F1 mean"]:.4f}'
        f1_sub = f'<div class="sd-td-sub">±{row["Fail F1 std"]:.4f}</div>'
        rec_val = f'{row["Fail Rec mean"]*100:.1f}%'
        prec_val = f'{row["Fail Prec mean"]*100:.1f}%'

        abl_rows.append(
            f'<tr{row_cls}>'
            f'<td>{name_html}</td>'
            f'<td class="sd-td-num">{prauc_main}{prauc_sub}</td>'
            f'<td class="sd-td-num">{f1_main}{f1_sub}</td>'
            f'<td class="sd-td-num">{rec_val}</td>'
            f'<td class="sd-td-num">{prec_val}</td>'
            f'</tr>'
        )
    abl_tbody = "".join(abl_rows)

    tables_html = (
        '<div id="holdout-test-set-performance-outputs-comparison-table-csv"></div>'
        '<div id="5-seed-validation-ablation-outputs-ablation-table-multiseed-csv"></div>'
        '<div class="sd-table-comparison-grid">'
        '<!-- Card 1: Holdout Test Set Performance -->'
        '<div class="sd-perf-card">'
        '<div class="sd-perf-card-header">'
        '<div class="sd-perf-card-topline">'
        '<div class="sd-perf-card-title">'
        '<span style="display:inline-block;width:8px;height:8px;background-color:#F51B0B;border-radius:1px;"></span>'
        'Holdout Test Set Performance'
        '</div>'
        '<span class="sd-perf-card-tag">Holdout Evaluation · 40 Wafers</span>'
        '</div>'
        '<div class="sd-perf-card-subtitle">'
        'Rigorous holdout evaluation on 39,351 unseen dies across 40 holdout wafers at tuned thresholds.'
        '</div>'
        '</div>'
        '<div class="sd-perf-table-wrapper">'
        '<table class="sd-perf-table">'
        '<thead>'
        '<tr>'
        '<th class="sd-th">Model</th>'
        '<th class="sd-th sd-th-num">Feats</th>'
        '<th class="sd-th sd-th-num">Fail F1</th>'
        '<th class="sd-th sd-th-num">PR-AUC</th>'
        '<th class="sd-th sd-th-num">Fail<br>Recall</th>'
        '<th class="sd-th sd-th-num">Fail<br>Prec</th>'
        '<th class="sd-th sd-th-num">Tuned<br>Thresh</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
        '<tr>'
        '<td>'
        '<span class="sd-badge-model-a">Model A</span>'
        '<div class="sd-model-sub">Die + Spatial</div>'
        '</td>'
        '<td class="sd-td-num">511</td>'
        f'<td class="sd-td-num">{f1_a:.4f}</td>'
        f'<td class="sd-td-num">{prauc_a:.4f}</td>'
        f'<td class="sd-td-num">{rec_a:.1f}%</td>'
        f'<td class="sd-td-num">{prec_a:.1f}%</td>'
        f'<td class="sd-td-num">{th_a:.4f}</td>'
        '</tr>'
        '<tr class="sd-row-highlight">'
        '<td>'
        '<span class="sd-badge-model-b">Model B</span>'
        '<div class="sd-model-sub" style="color:#C084FC;">+ Block Sub-Die</div>'
        '</td>'
        '<td class="sd-td-num" style="color:#C084FC;font-weight:700;">531</td>'
        f'<td class="sd-td-num">{f1_b:.4f}<div class="sd-td-sub-muted">unchanged</div></td>'
        f'<td class="sd-td-num"><span style="color:#43D17C;font-weight:700;">{prauc_b:.4f}</span><div class="sd-td-sub-pos">+{prauc_delta:.4f}</div></td>'
        f'<td class="sd-td-num">{rec_b:.1f}%<div class="sd-td-sub-pos">+{rec_delta:.1f}%</div></td>'
        f'<td class="sd-td-num">{prec_b:.1f}%<div class="sd-td-sub-neg">{prec_delta:.1f}%</div></td>'
        f'<td class="sd-td-num">{th_b:.4f}</td>'
        '</tr>'
        '</tbody>'
        '</table>'
        '</div>'
        '<div class="sd-perf-card-footer">'
        '<span style="color:#43D17C;font-weight:700;">✓</span>'
        '<span>PR-AUC improves +0.0337 (+6.7%) with +20 sub-die features while preserving 97.1% overall accuracy.</span>'
        '</div>'
        '</div>'
        '<!-- Card 2: 5-Seed Validation Ablation -->'
        '<div class="sd-perf-card">'
        '<div class="sd-perf-card-header">'
        '<div class="sd-perf-card-topline">'
        '<div class="sd-perf-card-title">'
        '<span style="display:inline-block;width:8px;height:8px;background-color:#A855F7;border-radius:1px;"></span>'
        '5-Seed Validation Ablation'
        '</div>'
        '<span class="sd-perf-card-tag">5-Seed Cross-Validation · 25 Folds</span>'
        '</div>'
        '<div class="sd-perf-card-subtitle">'
        'Stratified cross-validation across 5 random seeds (25 folds per configuration) measuring signal progression.'
        '</div>'
        '</div>'
        '<div class="sd-perf-table-wrapper">'
        '<table class="sd-perf-table">'
        '<thead>'
        '<tr>'
        '<th class="sd-th">Feature Configuration</th>'
        '<th class="sd-th sd-th-num">PR-AUC<span class="sd-th-sub">Mean ± Std</span></th>'
        '<th class="sd-th sd-th-num">Fail F1<span class="sd-th-sub">Mean ± Std</span></th>'
        '<th class="sd-th sd-th-num">Fail Recall<span class="sd-th-sub">Mean</span></th>'
        '<th class="sd-th sd-th-num">Fail Precision<span class="sd-th-sub">Mean</span></th>'
        '</tr>'
        '</thead>'
        f'<tbody>{abl_tbody}</tbody>'
        '</table>'
        '</div>'
        '<div class="sd-perf-card-footer">'
        '<span style="color:#A855F7;font-weight:700;">✓</span>'
        '<span>Sub-die block signals raise 5-seed validation PR-AUC from 0.4914 to 0.5265 (+0.0351 gain).</span>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(tables_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SIDEBAR: Control Center (design.md Section 15)
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 10px; height: 10px; background-color: #F51B0B; border-radius: 1px;"></span>
                DIEYIELD INTELLIGENCE
            </div>
            <div style="font-size: 0.78rem; color: #7F8792; margin-top: 2px; font-weight: 500;">
                Multi-Resolution Yield Prediction
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Demo Presets
        st.markdown('<div class="sd-kpi-label">DEMO MODE</div>', unsafe_allow_html=True)
        preset_names = [
            "Manual Exploration",
            "Preset 1: W_F_0014 (40,18) — Spatial Context True Fail",
            "Preset 2: W_N_0066 (2,13) — Block-driven B-only Catch",
            "Preset 3: W_F_0047 (4,10) — Second B-only Catch",
            "Preset 4: W_F_0016 (21,12) — Borderline False Alarm (Pass)",
        ]
        chosen_preset = st.selectbox(
            "Demo Presets",
            options=preset_names,
            index=0,
            key="preset_select",
            label_visibility="collapsed",
            help="Select one of the 4 benchmark demonstration dies."
        )

        preset_map = {
            "Preset 1: W_F_0014 (40,18) — Spatial Context True Fail": ("W_F_0014", 40, 18, "preset_1"),
            "Preset 2: W_N_0066 (2,13) — Block-driven B-only Catch": ("W_N_0066", 2, 13, "preset_2"),
            "Preset 3: W_F_0047 (4,10) — Second B-only Catch": ("W_F_0047", 4, 10, "preset_3"),
            "Preset 4: W_F_0016 (21,12) — Borderline False Alarm (Pass)": ("W_F_0016", 21, 12, "preset_4"),
        }

        preset_data = preset_map.get(chosen_preset)
        if preset_data is not None:
            preset_wafer, preset_row, preset_col, preset_id = preset_data
            wafer_default_idx = wafers.index(preset_wafer) if preset_wafer in wafers else 0
        else:
            preset_wafer, preset_row, preset_col, preset_id = None, None, None, "manual"
            wafer_default_idx = wafers.index("W_F_0014") if "W_F_0014" in wafers else 0

        # Wafer selection
        st.markdown('<div class="sd-kpi-label" style="margin-top: 14px;">WAFER</div>', unsafe_allow_html=True)
        selected_wafer = st.selectbox(
            "Select Wafer ID",
            options=wafers,
            index=wafer_default_idx,
            key=f"wafer_select_{preset_id}",
            label_visibility="collapsed",
            help="Choose a test wafer to inspect wafer spatial patterns and per-die diagnostics."
        )

        # Model mode
        st.markdown('<div class="sd-kpi-label" style="margin-top: 14px;">VIEW</div>', unsafe_allow_html=True)
        model_mode = st.radio(
            "Model Selection",
            options=["Model B (Full Diagnostic)", "Model A (Baseline + Spatial)", "Side-by-side Comparison"],
            index=0,
            key="model_mode_select",
            label_visibility="collapsed",
            help="Switch between Model A, Model B, or view both side-by-side."
        )

        st.markdown('<div class="sd-divider" style="margin: 20px 0 16px 0;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; color: #7F8792; text-transform: uppercase; margin-bottom: 8px;">
            DATASET
        </div>
        <div style="font-size: 0.8rem; color: #B7BDC7; line-height: 1.5; margin-bottom: 14px;">
            40 held-out wafers<br>
            32,598 eligible dies
        </div>
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; color: #7F8792; text-transform: uppercase; margin-bottom: 8px;">
            MODEL
        </div>
        <div style="font-size: 0.8rem; color: #B7BDC7; line-height: 1.5; margin-bottom: 12px;">
            Model A: 511 features<br>
            Model B: 531 features
        </div>
        <div>
            <span class="sd-badge-verified">Evaluation verified</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sd-footer">
            <span>DieYield Intelligence • v1.0</span>
        </div>
        """, unsafe_allow_html=True)

    # Slice data for selected wafer
    w_mask = meta_df["wafer_id"] == selected_wafer
    w_indices = np.where(w_mask)[0]
    w_meta = meta_df.iloc[w_indices].reset_index(drop=True)
    w_prob_a = prob_a[w_indices]
    w_prob_b = prob_b[w_indices]
    w_prob_a_cal = prob_a_cal[w_indices] if prob_a_cal is not None else None
    w_prob_b_cal = prob_b_cal[w_indices] if prob_b_cal is not None else None

    n_w_dies = len(w_meta)
    n_w_old_fails = int((w_meta["old_label"] == 1).sum())
    n_w_eligible = int((w_meta["old_label"] == 0).sum())
    has_gt = "label" in w_meta.columns
    n_w_true_new_fails = int((w_meta.loc[w_meta["old_label"] == 0, "label"] == 1).sum()) if has_gt else None

    # Overview KPI Cards
    st.markdown('<div class="sd-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sd-section-title">Wafer Spatial Map &amp; Risk Fields (<span style="font-family: \'JetBrains Mono\'; color: var(--sd-model-a);">{selected_wafer}</span>)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-sub">Inspect full-wafer spatial signature, pre-test failure masks, model probability heatmaps, and continuous 2D Gaussian risk contours.</div>', unsafe_allow_html=True)

    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    with col_kpi1:
        st.markdown(f'<div class="sd-card"><div class="sd-kpi-label">Selected Wafer</div><div class="sd-kpi-value" style="font-size: 1.35rem; font-family: \'JetBrains Mono\';">{selected_wafer}</div></div>', unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f'<div class="sd-card"><div class="sd-kpi-label">Total Dies</div><div class="sd-kpi-value">{n_w_dies:,}</div></div>', unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f'<div class="sd-card"><div class="sd-kpi-label">Pre-Test Fails</div><div class="sd-kpi-value">{n_w_old_fails:,}</div></div>', unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f'<div class="sd-card"><div class="sd-kpi-label">Eligible Dies</div><div class="sd-kpi-value">{n_w_eligible:,}</div></div>', unsafe_allow_html=True)
    with col_kpi5:
        gt_display = f"{n_w_true_new_fails:,}" if has_gt else "N/A"
        st.markdown(f'<div class="sd-card"><div class="sd-kpi-label">Actual New Fails</div><div class="sd-kpi-value" style="color: var(--sd-failure);">{gt_display}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 3: 4-Panel Wafer View (design.md Section 18)
    # -------------------------------------------------------------------------
    if model_mode == "Side-by-side Comparison":
        st.markdown("#### Model A (Die + Spatial + Die Anomaly)")
        fig_a = render_wafer_4panel(
            w_meta, w_prob_a, meta_a["threshold"], selected_wafer, "Model A", has_ground_truth=has_gt
        )
        st.pyplot(fig_a)
        plt.close(fig_a)

        st.markdown("#### Model B (Model A + Block Summary & Anomaly)")
        fig_b = render_wafer_4panel(
            w_meta, w_prob_b, meta_b["threshold"], selected_wafer, "Model B", has_ground_truth=has_gt
        )
        st.pyplot(fig_b)
        plt.close(fig_b)
    elif model_mode == "Model A (Baseline + Spatial)":
        fig = render_wafer_4panel(
            w_meta, w_prob_a, meta_a["threshold"], selected_wafer, "Model A", has_ground_truth=has_gt
        )
        st.pyplot(fig)
        plt.close(fig)
    else:  # Model B
        fig = render_wafer_4panel(
            w_meta, w_prob_b, meta_b["threshold"], selected_wafer, "Model B", has_ground_truth=has_gt
        )
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # SECTION 4: Die Selector & Detailed Explanation Panel
    # -------------------------------------------------------------------------
    st.markdown('<div class="sd-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-title">Die-Level Diagnostic Inspector &amp; Decision Strip</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-sub">Inspect individual die predictions, uncalibrated raw scores, calibrated failure risks, and model decision thresholds.</div>', unsafe_allow_html=True)

    # Determine active model parameters for die inspection
    active_is_model_b = ("Model B" in model_mode) or (model_mode == "Side-by-side Comparison")
    active_model_name = "Model B" if active_is_model_b else "Model A"
    active_probs = w_prob_b if active_is_model_b else w_prob_a
    active_threshold = meta_b["threshold"] if active_is_model_b else meta_a["threshold"]
    active_model = model_b if active_is_model_b else model_a
    active_pass_med = pass_med_b if active_is_model_b else pass_med_a

    # Build die options list
    w_meta["_prob"] = active_probs
    w_meta["_pred"] = (active_probs >= active_threshold).astype(int)
    w_meta.loc[w_meta["old_label"] == 1, "_pred"] = 1

    # Check if this wafer contains any benchmark dies
    wafer_reps = [coord for coord in PRECOMPUTED_REPRESENTATIVES.keys() if coord[0] == selected_wafer]

    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        # Pre-select benchmark die if present, else highest risk eligible die
        top_risk_dies = w_meta.sort_values("_prob", ascending=False)
        die_coord_options = [
            f"Row {int(r)}, Col {int(c)} (Score: {p*100:.1f}%, Status: {'FAIL' if pr==1 else 'PASS'})"
            for r, c, p, pr in zip(top_risk_dies["die_row"], top_risk_dies["die_col"], top_risk_dies["_prob"], top_risk_dies["_pred"])
        ]

        # Check default index
        default_die_idx = 0
        if preset_row is not None and preset_col is not None and selected_wafer == preset_wafer:
            for i, opt in enumerate(die_coord_options):
                if f"Row {preset_row}, Col {preset_col} " in opt:
                    default_die_idx = i
                    break
        elif wafer_reps:
            rep_r, rep_c = wafer_reps[0][1], wafer_reps[0][2]
            for i, opt in enumerate(die_coord_options):
                if f"Row {rep_r}, Col {rep_c} " in opt:
                    default_die_idx = i
                    break

        selected_die_str = st.selectbox(
            "Select Die to Inspect (Sorted by Failure Risk Score)",
            options=die_coord_options,
            index=default_die_idx,
            key=f"die_{selected_wafer}_{preset_id}",
            help="Select any die on the wafer grid to inspect local feature SHAP attributions and sub-die profiles."
        )

        # Parse selected coordinates
        import re
        match = re.search(r"Row (\d+), Col (\d+)", selected_die_str)
        sel_row = int(match.group(1))
        sel_col = int(match.group(2))

    # Retrieve selected die metadata and local index
    sel_local_idx = int(w_meta[(w_meta["die_row"] == sel_row) & (w_meta["die_col"] == sel_col)].index[0])
    sel_global_idx = int(w_indices[sel_local_idx])
    sel_die_row = w_meta.iloc[sel_local_idx]

    sel_prob = float(active_probs[sel_local_idx])
    sel_pred = int(w_meta.iloc[sel_local_idx]["_pred"])
    sel_old_label = int(sel_die_row["old_label"])
    sel_gt = int(sel_die_row["label"]) if has_gt else None

    with col_sel2:
        st.markdown(f"**Die Coordinates: ({sel_row}, {sel_col}) on Wafer `{selected_wafer}`**")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        p_a_die = float(w_prob_a[sel_local_idx])
        p_b_die = float(w_prob_b[sel_local_idx])
        p_a_die_cal = float(w_prob_a_cal[sel_local_idx]) if w_prob_a_cal is not None else None
        p_b_die_cal = float(w_prob_b_cal[sel_local_idx]) if w_prob_b_cal is not None else None
        active_cal_prob = p_b_die_cal if active_is_model_b else p_a_die_cal
        
        with col_m1:
            cal_delta = f"Calibrated: {active_cal_prob*100:.1f}%" if active_cal_prob is not None else None
            st.metric("Raw Model Score", f"{sel_prob*100:.1f}%", delta=cal_delta)
        with col_m2:
            st.metric("Decision Threshold", f"{active_threshold:.4f}")
        with col_m3:
            st.metric("Pre-Test Status", "FAIL (old_label=1)" if sel_old_label == 1 else "PASS (Eligible)")
        with col_m4:
            gt_text = ("FAIL" if sel_gt == 1 else "PASS") if has_gt else "N/A"
            st.metric("Ground-Truth Target", gt_text)

        # A-vs-B Decision Strip
        dec_a = (p_a_die >= meta_a["threshold"]) if sel_old_label == 0 else True
        dec_b = (p_b_die >= meta_b["threshold"]) if sel_old_label == 0 else True
        diff_pct = (p_b_die - p_a_die) * 100

        st.markdown("---")
        st.markdown("**Model A vs. Model B Head-to-Head Decision Strip**")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            cal_note_a = f" (cal: {p_a_die_cal*100:.1f}%)" if p_a_die_cal is not None else ""
            st.markdown(f"**Model A**: `{p_a_die*100:.1f}%`{cal_note_a} → **{'FAIL' if dec_a else 'PASS'}** (cutoff: `{meta_a['threshold']:.4f}`)")
        with col_s2:
            cal_note_b = f" (cal: {p_b_die_cal*100:.1f}%)" if p_b_die_cal is not None else ""
            st.markdown(f"**Model B**: `{p_b_die*100:.1f}%`{cal_note_b} → **{'FAIL' if dec_b else 'PASS'}** (cutoff: `{meta_b['threshold']:.4f}`)")
        with col_s3:
            st.markdown(f"**Shift (B − A)**: `{diff_pct:+.1f}%` ({'Both Agree' if dec_a == dec_b else 'Decision Diverges'})")

    # Spacer gives the die-selector dropdown clearance when open,
    # so it does not visually intersect the next section.
    st.markdown('<div style="padding-bottom: 200px; margin-bottom: -200px;"></div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 5: Hero Demo — Why Did Model B Change Its Mind? (design.md Section 19)
    # -------------------------------------------------------------------------
    st.markdown('<div class="sd-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-title">Why Did Model B Change Its Mind?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-sub">Sub-die evidence reveals localized variation that die-level averages can hide.</div>', unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns([1, 1])
    with col_exp1:
        st.markdown("""
        <div class="sd-hero-die-card">
            <div style="font-weight: 800; color: #FFFFFF; font-size: 1.1rem; margin-bottom: 2px;">
                Benchmark Case: Wafer W_N_0066 · Die (2, 13)
            </div>
            <div style="font-size: 0.8rem; color: #7F8792; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.05em;">
                True post-burn-in defect missed by die-level &amp; spatial features
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #0D0F12; border: 1px solid #30353D; border-radius: 6px; margin-bottom: 10px;">
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #25B9E6; text-transform: uppercase;">MODEL A</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">12.9%</div>
                    <div style="font-size: 0.75rem; color: #7F8792;">Cal: 4.5%</div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #35D07F; margin-top: 2px;">PASS</div>
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #7F8792; padding: 0 10px;">→</div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #A855F7; text-transform: uppercase;">MODEL B</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">62.9%</div>
                    <div style="font-size: 0.75rem; color: #7F8792;">Cal: 45.2%</div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #FF5A5F; margin-top: 2px;">FLAGGED</div>
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #7F8792; padding: 0 10px;">→</div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #FF5A5F; text-transform: uppercase;">GROUND TRUTH</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #FF5A5F;">FAIL</div>
                    <div style="font-size: 0.75rem; color: #7F8792;">old_label: 0</div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: #FF5A5F; margin-top: 2px;">TRUE DEFECT</div>
                </div>
            </div>
            <div style="font-size: 0.84rem; color: #B7BDC7; line-height: 1.5; padding-top: 6px;">
                <b>The Multi-Resolution Difference:</b> Model A sees the die-level average (12.9% raw score, well below cutoff 0.5574). Model B detects localized sub-die variation (<code>blk_anomaly_score = 0.594</code> across 2,000 block readings), pushing the score to 62.9% and catching the failure early.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_exp2:
        st.markdown("""
        <div class="sd-card">
            <div style="font-weight: 800; color: #A855F7; font-size: 1.1rem; margin-bottom: 2px;">
                Discrepancy Impact Across 40 Test Wafers
            </div>
            <div style="font-size: 0.8rem; color: #7F8792; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.05em;">
                Net Defect Catch vs. False Scrap Trade-off
            </div>
            <div style="font-size: 0.84rem; color: #B7BDC7; line-height: 1.6;">
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #30353D;">
                    <span>Failures caught by Model B only (Model A missed):</span>
                    <span style="font-weight: 700; color: #35D07F;">+29 fails</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #30353D;">
                    <span>Failures caught by Model A only (Model B missed):</span>
                    <span style="font-weight: 700; color: #FF5A5F;">-8 fails</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #30353D;">
                    <span><b>Net True Defect Gain (at tuned cutoffs):</b></span>
                    <span style="font-weight: 800; color: #FFFFFF;">+21 net defects caught</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                    <span>False alarm dies (scrap trade-off):</span>
                    <span style="color: #7F8792;">69 on Model B vs. 15 on Model A</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Discrepancy table
    el_all = (meta_df["old_label"] == 0)
    y_all = meta_df.loc[el_all, "label"].values if "label" in meta_df else np.zeros(el_all.sum())
    pa_all = prob_a[el_all.values]
    pb_all = prob_b[el_all.values]
    th_a = meta_a["threshold"]
    th_b = meta_b["threshold"]

    gained_mask = (y_all == 1) & (pa_all < th_a) & (pb_all >= th_b)
    gained_dies = meta_df[el_all][gained_mask][["wafer_id", "die_row", "die_col"]].copy()
    gained_dies["Model A Risk"] = [f"{p*100:.1f}%" for p in pa_all[gained_mask]]
    gained_dies["Model B Risk"] = [f"{p*100:.1f}%" for p in pb_all[gained_mask]]
    gained_dies["Risk Shift (B − A)"] = [f"{(b - a)*100:+.1f}%" for a, b in zip(pa_all[gained_mask], pb_all[gained_mask])]
    gained_dies["Status"] = "Caught by Model B Only (True Post-Test Fail)"

    st.markdown(f"#### :material/search: Complete Discrepancy Table: 29 True Failures Caught by Model B Only")
    st.dataframe(gained_dies.reset_index(drop=True), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # SECTION 6: Diagnostic Explainability & Sub-Die Signal Profiling (design.md Section 21)
    # -------------------------------------------------------------------------
    st.markdown('<div class="sd-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-title">Diagnostic Explainability: Why Was This Die Flagged?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-sub">Deconstruct local model predictions into parametric, spatial, and sub-die block feature contributions using TreeSHAP.</div>', unsafe_allow_html=True)

    # Compute SHAP values for this wafer (cached on wafer selection)
    shap_vals, feat_cols, X_wafer = compute_wafer_shap(selected_wafer, model_type=active_model_name)
    die_shap = shap_vals[sel_local_idx]
    die_X = X_wafer.iloc[sel_local_idx]

    # Sort features by absolute SHAP attribution
    feat_order = np.argsort(np.abs(die_shap))[::-1]
    top_n = 8
    top_indices = feat_order[:top_n]
    top_names = [feat_cols[i] for i in top_indices]
    top_shaps = [die_shap[i] for i in top_indices]
    top_vals = [die_X[feat_cols[i]] for i in top_indices]

    # Top 3 Driver Cards (design.md Section 21)
    st.markdown("#### Primary Prediction Drivers")
    col_d1, col_d2, col_d3 = st.columns(3)
    driver_cols = [col_d1, col_d2, col_d3]
    for i in range(min(3, len(top_names))):
        fname = top_names[i]
        fval = top_vals[i]
        fshap = top_shaps[i]
        is_pos = (fshap > 0)
        direction_icon = "↑ increases risk" if is_pos else "↓ reduces risk"
        direction_color = "var(--sd-failure)" if is_pos else "var(--sd-success)"
        context_tag = ' <span class="sd-badge-context">Context</span>' if fname.startswith("sp_") else ''
        
        with driver_cols[i]:
            st.markdown(f"""
            <div class="sd-card" style="padding: 14px 16px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #7F8792; text-transform: uppercase;">DRIVER #{i+1}{context_tag}</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{fname}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
                    <span style="color: #B7BDC7;">Value: {fval:.3g}</span>
                    <span style="font-weight: 700; color: {direction_color};">{direction_icon}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Display SHAP breakdown & Domain attribution
    col_shap1, col_shap2 = st.columns([3, 2])
    with col_shap1:
        st.markdown(f"#### Top Feature Attributions ({active_model_name})")

        # Horizontal bar chart of top features (SanDisk styling)
        fig_bar, ax_bar = plt.subplots(figsize=(7, 3.8), facecolor="#15181D")
        ax_bar.set_facecolor("#0D0F12")
        ax_bar.tick_params(colors="#7F8792", labelsize=8)
        for spine in ax_bar.spines.values():
            spine.set_color("#30353D")

        y_positions = np.arange(top_n)[::-1]
        colors = ["#FF5A5F" if s > 0 else "#35D07F" for s in top_shaps]
        ax_bar.barh(y_positions, top_shaps, color=colors, height=0.6)
        ax_bar.axvline(0, color="#7F8792", linewidth=0.8, linestyle="--")

        formatted_labels = [
            f"{name} ({val:.2g}) (context, not an intervention)" if name.startswith("sp_")
            else f"{name} ({val:.2g})"
            for name, val in zip(top_names, top_vals)
        ]
        ax_bar.set_yticks(y_positions)
        ax_bar.set_yticklabels(formatted_labels, color="#FFFFFF", fontsize=8)
        ax_bar.set_xlabel("SHAP Attribution (Red = Increases Failure Risk, Green = Reduces Risk)", color="#B7BDC7", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with col_shap2:
        st.markdown("#### Domain Attribution Breakdown")
        
        # Domain contributions
        die_contrib = float(sum(abs(die_shap[i]) for i, f in enumerate(feat_cols) if f.startswith("feature_")))
        sp_contrib = float(sum(abs(die_shap[i]) for i, f in enumerate(feat_cols) if f.startswith("sp_")))
        blk_contrib = float(sum(abs(die_shap[i]) for i, f in enumerate(feat_cols) if f.startswith("blk_")))
        anom_contrib = float(sum(abs(die_shap[i]) for i, f in enumerate(feat_cols) if "anomaly" in f))
        total_mass = max(die_contrib + sp_contrib + blk_contrib + anom_contrib, 1e-9)

        domain_df = pd.DataFrame({
            "Domain": [
                "Die Parametric (500)",
                "Spatial Neighborhood (10) (Context — not an intervention)",
                "Sub-Die Block (19)",
                "Anomaly Scores (1-2)"
            ],
            "Role": [
                "Primary signal",
                "Context (not an intervention)",
                "Micro-structural evidence",
                "Supporting signal"
            ],
            "Percentage": [
                f"{die_contrib/total_mass*100:.1f}%",
                f"{sp_contrib/total_mass*100:.1f}%",
                f"{blk_contrib/total_mass*100:.1f}%",
                f"{anom_contrib/total_mass*100:.1f}%",
            ]
        })
        st.dataframe(domain_df, use_container_width=True, hide_index=True)

        # Failure Signature matching
        st.markdown("#### Signature (Rule-Based on Top SHAP Driver)")
        if sel_pred == 1 or sel_prob >= active_threshold:
            top_f = top_names[0]
            if "blk_" in top_f:
                sig_text = "Cluster 0 / 1: Block-Reading Signal Drift / Memory Array Shift"
                sig_desc = "Driven by sub-die block voltage anomalies indicating local memory array degradation."
            elif "sp_dist" in top_f or "sp_old" in top_f:
                sig_text = "Cluster 2 / 3: Defect Neighborhood Proximity & Spatial Clustering"
                sig_desc = "Driven by physical proximity to existing pre-test wafer defect clusters."
            else:
                sig_text = "Cluster -1: Mixed / Parametric Electrical Breakdown"
                sig_desc = "Multi-parametric electrical shift across die-level measurements."
            
            st.info(f"**{sig_text}**\n\n_{sig_desc}_")
        else:
            st.success("Die is predicted as **PASS** (Normal operating population).")

    # Counterfactual explanation box (design.md Section 23)
    st.markdown("#### Risk Sensitivity — What Drives the Score?")
    st.markdown('<div class="sd-section-sub">If measurable signals move toward their normal range, how does the model score respond?</div>', unsafe_allow_html=True)
    rep_key = (selected_wafer, sel_row, sel_col)
    if rep_key in PRECOMPUTED_REPRESENTATIVES:
        rep_info = PRECOMPUTED_REPRESENTATIVES[rep_key]
        st.markdown(f"**Representative Benchmark Die Identified**: `{rep_info['title']}`")
        st.markdown(f"""
        - **Model Confidence**: {rep_info['confidence']}
        - **Failure Signature**: {rep_info['cluster']}
        - **Stepwise Normalization Trajectory**:  
          `{rep_info['trajectory']}`
        - **Total Achievable Risk Reduction**: **{rep_info['reduction']}**
        """)
    else:
        st.markdown("""
        <div class="sd-disclaimer">
            <b>Note on Counterfactual Sensitivity</b>: Full 5-model bootstrap ensemble uncertainty trajectories were precomputed for benchmark cases (e.g. Wafer <code>W_F_0014</code> Row 40, Col 18; Wafer <code>W_F_0019</code> Row 22, Col 22; Wafer <code>W_F_0016</code> Row 21, Col 12).
        </div>
        """, unsafe_allow_html=True)
        
        # Lightweight single-feature live counterfactual re-score (using top non-spatial driver)
        non_spatial_drivers = [f for f in top_names if not f.startswith("sp_")]
        top_driver = non_spatial_drivers[0] if non_spatial_drivers else top_names[0]
        if top_driver in active_pass_med:
            die_vec_cf = die_X.copy().to_frame().T
            die_vec_cf[top_driver] = active_pass_med[top_driver]
            p_cf = float(active_model.predict_proba(die_vec_cf[feat_cols].values)[0, 1])
            delta_p = sel_prob - p_cf
            st.markdown(f"""
            - **Live Sensitivity Test**: Normalizing top non-spatial driver `{top_driver}` from `{die_X[top_driver]:.3g}` to healthy median `{active_pass_med[top_driver]:.3g}` shifts failure probability from **{sel_prob*100:.1f}%** to **{p_cf*100:.1f}%** (Δ = {delta_p*100:+.1f}%).
            """)

    st.caption(":material/warning: **Disclaimer**: Model-based mathematical sensitivity estimate; not a physical semiconductor manufacturing simulation or causal intervention.")

    # Sub-Die Block View (Model B Only, design.md Section 24 & 25)
    if active_is_model_b:
        st.markdown('<div class="sd-divider" style="margin: 24px 0 16px 0;"></div>', unsafe_allow_html=True)
        st.markdown("#### Sub-Die Evidence — 2,000 Internal Readings")
        
        w_blk_df = load_wafer_block_readings(selected_wafer)
        die_blk_row = w_blk_df[
            (w_blk_df["die_row"] == sel_row) &
            (w_blk_df["die_col"] == sel_col)
        ]

        if not die_blk_row.empty:
            raw_str = die_blk_row.iloc[0]["block_readings"]
            readings_arr = np.fromstring(raw_str, sep=" ")
            
            fig_strip = render_block_strip(readings_arr, sel_row, sel_col, selected_wafer)
            st.pyplot(fig_strip)
            plt.close(fig_strip)
            
            st.caption(
                ":material/push_pin: **Physical Mapping & Threshold Notice**: Anomalous points are highlighted using the robust MAD threshold from `src/block_features.py` (|reading - median| > 2.0 × MAD). "
                "The X-axis indicates sequential index position within the stream (0..1999) — NOT genuine physical 2D/3D spatial coordinates within the die stack."
            )
            st.markdown("""
            <div class="sd-disclaimer" style="margin-top: 10px;">
                <b>Feature audit</b>: Four block statistics were non-contributory in this experiment (<code>blk_n_anom_fixed</code>, <code>blk_frac_anom_fixed</code>, <code>blk_longest_run</code>, <code>blk_n_clusters</code>). Central tendency (mean), spread (std), and quantile statistics carry the signal. They remain in the model definition for reproducibility.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Block reading stream for Die ({sel_row}, {sel_col}) is not available in test store.")

    # -------------------------------------------------------------------------
    # SECTION 7: Budget-Aware Screening Triage (design.md Section 26 & 27)
    # -------------------------------------------------------------------------
    st.markdown('<div class="sd-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-title">Same Inspection Budget. More Failures Caught.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-sub">At the same inspection budget, prioritizing dies by continuous risk score catches more defective dies than baseline models.</div>', unsafe_allow_html=True)

    col_tr_ctrl1, col_tr_ctrl2 = st.columns([1, 1])
    with col_tr_ctrl1:
        triage_scope = st.radio(
            "Evaluation Scope",
            options=["All Test Wafers (40 Wafers, 32,598 Eligible Dies)", f"Selected Wafer ({selected_wafer})"],
            horizontal=True,
            key="triage_scope_select"
        )
    with col_tr_ctrl2:
        quick_pick = st.radio(
            "Quick Inspection Budget Presets",
            options=["5.0% (Default)", "1.0%", "2.0%", "10.0%", "Custom Slider"],
            horizontal=True,
            index=0,
            key="triage_quick_pick"
        )

    if quick_pick == "1.0%":
        screen_k_pct = 1.0
    elif quick_pick == "2.0%":
        screen_k_pct = 2.0
    elif quick_pick == "5.0% (Default)":
        screen_k_pct = 5.0
    elif quick_pick == "10.0%":
        screen_k_pct = 10.0
    else:
        screen_k_pct = st.slider(
            "Screen top K% of eligible dies",
            min_value=0.5,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="triage_k_slider"
        )

    # Determine slice
    use_cal = (prob_a_cal is not None and prob_b_cal is not None)
    if triage_scope.startswith("All Test Wafers"):
        el_mask = (meta_df["old_label"] == 0).values
        y_scope = meta_df.loc[el_mask, "label"].values if "label" in meta_df else np.zeros(el_mask.sum())
        s_a = prob_a[el_mask]
        s_b = prob_b[el_mask]
        s_a_cal = prob_a_cal[el_mask] if use_cal else None
        s_b_cal = prob_b_cal[el_mask] if use_cal else None
    else:
        el_mask = (w_meta["old_label"] == 0).values
        y_scope = w_meta.loc[el_mask, "label"].values if "label" in w_meta else np.zeros(el_mask.sum())
        s_a = w_prob_a[el_mask]
        s_b = w_prob_b[el_mask]
        s_a_cal = w_prob_a_cal[el_mask] if use_cal else None
        s_b_cal = w_prob_b_cal[el_mask] if use_cal else None

    # Compute metrics
    stats_a = topk_stats(y_scope, s_a, screen_k_pct / 100.0, score_cal=s_a_cal)
    stats_b = topk_stats(y_scope, s_b, screen_k_pct / 100.0, score_cal=s_b_cal)

    # Side-by-side comparison cards (Unified HTML cards — No empty boxes, zero truncation)
    score_label = "Mean Calibrated Risk" if use_cal else "Mean Risk Score"

    cal_rate_a_str = f"{stats_a['mean_calibrated_rate']*100:.1f}%" if stats_a['mean_calibrated_rate'] is not None else f"{stats_a['mean_score']:.3f}"
    cal_rate_b_str = f"{stats_b['mean_calibrated_rate']*100:.1f}%" if stats_b['mean_calibrated_rate'] is not None else f"{stats_b['mean_score']:.3f}"

    diff_fails = stats_b['n_fails_captured'] - stats_a['n_fails_captured']
    diff_cap = (stats_b['capture_rate'] - stats_a['capture_rate']) * 100
    diff_prec = (stats_b['observed_fail_rate'] - stats_a['observed_fail_rate']) * 100
    diff_lift = stats_b['lift'] - stats_a['lift']

    delta_fails_cls = "sd-triage-delta-pos" if diff_fails > 0 else "sd-triage-delta-neutral"
    delta_cap_cls = "sd-triage-delta-pos" if diff_cap > 0 else "sd-triage-delta-neutral"
    delta_prec_cls = "sd-triage-delta-pos" if diff_prec >= 0 else "sd-triage-delta-neutral"
    delta_lift_cls = "sd-triage-delta-pos" if diff_lift >= 0 else "sd-triage-delta-neutral"

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"""
        <div class="sd-triage-card" style="border-left: 4px solid var(--sd-model-a);">
            <div class="sd-triage-header">
                <div class="sd-triage-title">
                    <span style="display:inline-block; width:8px; height:8px; background:var(--sd-model-a); border-radius:1px;"></span>
                    Model A (Spatial Baseline)
                </div>
                <span class="sd-badge-model-a">Baseline · 511 Feats</span>
            </div>
            <div class="sd-triage-grid">
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Dies Screened</div>
                    <div class="sd-triage-val-compact font-mono">{stats_a['n_screened']:,} <span class="sd-triage-sub">({screen_k_pct:.1f}%)</span></div>
                    <div class="sd-triage-delta sd-triage-delta-neutral">Budget: {screen_k_pct:.1f}%</div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Failures Captured</div>
                    <div class="sd-triage-val-compact font-mono">{stats_a['n_fails_captured']:,} <span class="sd-triage-sub">/ {stats_a['n_fails_total']:,}</span></div>
                    <div class="sd-triage-delta sd-triage-delta-neutral">True defect dies</div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Capture Rate</div>
                    <div class="sd-triage-val font-mono" style="color: var(--sd-model-a);">{stats_a['capture_rate']*100:.1f}%</div>
                    <div class="sd-triage-delta sd-triage-delta-neutral">Of all failures</div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Failure Rate (Precision)</div>
                    <div class="sd-triage-val font-mono">{stats_a['observed_fail_rate']*100:.1f}%</div>
                    <div class="sd-triage-delta sd-triage-delta-neutral">Screened precision</div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Screening Lift</div>
                    <div class="sd-triage-val font-mono">{stats_a['lift']:.2f}×</div>
                    <div class="sd-triage-delta sd-triage-delta-neutral">vs Random (1.0×)</div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">{score_label}</div>
                    <div class="sd-triage-val font-mono">{cal_rate_a_str}</div>
                    <div class="sd-triage-delta sd-triage-delta-neutral">{'Calibrated risk' if use_cal else 'Raw score'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown(f"""
        <div class="sd-triage-card" style="border-left: 4px solid var(--sd-model-b);">
            <div class="sd-triage-header">
                <div class="sd-triage-title">
                    <span style="display:inline-block; width:8px; height:8px; background:var(--sd-model-b); border-radius:1px;"></span>
                    Model B (Multi-Resolution)
                </div>
                <span class="sd-badge-model-b">Champion · 531 Feats</span>
            </div>
            <div class="sd-triage-grid">
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Dies Screened</div>
                    <div class="sd-triage-val-compact font-mono">{stats_b['n_screened']:,} <span class="sd-triage-sub">({screen_k_pct:.1f}%)</span></div>
                    <div class="sd-triage-delta sd-triage-delta-neutral">Same budget</div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Failures Captured</div>
                    <div class="sd-triage-val-compact font-mono">{stats_b['n_fails_captured']:,} <span class="sd-triage-sub">/ {stats_b['n_fails_total']:,}</span></div>
                    <div class="sd-triage-delta {delta_fails_cls}"><strong>{diff_fails:+d} dies</strong></div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Capture Rate</div>
                    <div class="sd-triage-val font-mono" style="color: var(--sd-model-b);">{stats_b['capture_rate']*100:.1f}%</div>
                    <div class="sd-triage-delta {delta_cap_cls}"><strong>{diff_cap:+.1f}%</strong></div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Failure Rate (Precision)</div>
                    <div class="sd-triage-val font-mono">{stats_b['observed_fail_rate']*100:.1f}%</div>
                    <div class="sd-triage-delta {delta_prec_cls}"><strong>{diff_prec:+.1f}%</strong></div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">Screening Lift</div>
                    <div class="sd-triage-val font-mono">{stats_b['lift']:.2f}×</div>
                    <div class="sd-triage-delta {delta_lift_cls}"><strong>{diff_lift:+.2f}×</strong></div>
                </div>
                <div class="sd-triage-cell">
                    <div class="sd-triage-label">{score_label}</div>
                    <div class="sd-triage-val font-mono">{cal_rate_b_str}</div>
                    <div class="sd-triage-delta sd-triage-delta-neutral">{'Calibrated risk' if use_cal else 'Raw score'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


    # Render Gains Curve (design.md Section 27)
    k_pts_a, cap_curve_a, _ = capture_curve(y_scope, s_a, n_points=200)
    k_pts_b, cap_curve_b, _ = capture_curve(y_scope, s_b, n_points=200)

    fig_gain, ax_gain = plt.subplots(figsize=(10, 4.2), facecolor="#15181D")
    ax_gain.set_facecolor("#0D0F12")
    ax_gain.plot([0, 100], [0, 100], "--", color="#7F8792", linewidth=1.1, label="Random Screening Baseline (Lift = 1.0×)")
    ax_gain.plot(k_pts_a * 100, cap_curve_a * 100, color="#25B9E6", linewidth=2.0, label="Model A (Spatial Baseline)")
    ax_gain.plot(k_pts_b * 100, cap_curve_b * 100, color="#A855F7", linewidth=2.2, label="Model B (Multi-Resolution)")

    ax_gain.axvline(screen_k_pct, color="#F51B0B", linestyle=":", linewidth=1.6, label=f"Current Budget: {screen_k_pct:.1f}%")
    ax_gain.scatter([screen_k_pct], [stats_a['capture_rate'] * 100], color="#25B9E6", s=45, zorder=5)
    ax_gain.scatter([screen_k_pct], [stats_b['capture_rate'] * 100], color="#A855F7", s=45, zorder=5)

    ax_gain.set_xlim(0, 20)
    ax_gain.set_ylim(0, 80)
    ax_gain.set_xlabel("Screening Budget (% of Eligible Dies Inspected)", color="#B7BDC7", fontsize=9)
    ax_gain.set_ylabel("Defect Capture Rate (% of True Failures Caught)", color="#B7BDC7", fontsize=9)
    ax_gain.set_title("Screening Gains Curve — Defect Capture vs. Inspection Budget", color="#FFFFFF", fontsize=10, pad=8, fontweight="bold")
    ax_gain.tick_params(colors="#7F8792", labelsize=8)
    for spine in ax_gain.spines.values():
        spine.set_color("#30353D")
    ax_gain.grid(True, linestyle=":", alpha=0.4, color="#30353D")
    ax_gain.legend(facecolor="#15181D", edgecolor="#30353D", labelcolor="#FFFFFF", fontsize=8, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_gain)
    plt.close(fig_gain)

    # -------------------------------------------------------------------------
    # SECTION 8: Methodological Limitations & Engineering Disclosures (design.md Section 43)
    # -------------------------------------------------------------------------
    st.markdown('<div class="sd-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-section-title">Methodological Limitations &amp; Engineering Disclosures</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sd-limitations">
        <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 8px;">
            Engineering Honesty &amp; Scope Disclosures:
        </div>
        <ol style="margin-top: 4px; margin-bottom: 4px; padding-left: 20px; line-height: 1.65; color: #B7BDC7;">
            <li><b>Semi-Synthetic Data Origin</b>: Real WM-811K wafer geometry and pre-test maps were combined with synthetic 500 parametric features, synthetic 2,000 block readings per die, and controlled post-test failure labels.</li>
            <li><b>Marginal Defect Distribution Overlap</b>: A large fraction of failures (65% in this semi-synthetic design) are marginal and overlap healthy electrical distributions, making binary classification difficult; observed F1 remains around 0.52.</li>
            <li><b>Ranking vs. Binary Classification</b>: Model B provides statistically meaningful improvements in continuous risk ranking (PR-AUC +0.034, 95% wafer-cluster bootstrap CI [+0.023, +0.043] strictly excludes zero) and screening triage (at the same 10% inspection budget, Model B captures 66 more failures than Model A: 834 vs. 768), but binary decision F1 difference (+0.001, 95% CI [−0.008, +0.012]) is not distinguishable from noise.</li>
            <li><b>Inert Block Features</b>: Central tendency (mean), spread (std), and quantile statistics carry the sub-die signal; sequential run and cluster statistics (<code>blk_n_anom_fixed</code>, <code>blk_longest_run</code>) were inert in this experimental setting.</li>
            <li><b>Spatial Context is Non-Interventionist</b>: Spatial features (<code>sp_dist_to_fail</code>, <code>sp_old_fail_density_5</code>) provide physical neighborhood context, not manufacturing interventions. They cannot be directly manipulated on a silicon wafer.</li>
            <li><b>Not Ready for Direct Fab Deployment</b>: Demonstrates multi-resolution proof of concept; full fab qualification requires retraining on unredacted, physical automatic test equipment (ATE) data streams.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Product Footer (design.md Section 44)
    st.markdown("""
    <div class="sd-footer">
        <span>DieYield Intelligence • Multi-Resolution Yield Prediction</span>
        <span>Model evaluation • v1.0</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
