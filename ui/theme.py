import streamlit as st


def apply_theme():
    """Apply global Streamlit theme & CSS. Keep deterministic (no network calls)."""
    st.set_page_config(
        page_title="🔍 Sentient Welfare Platform 2026",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root {
            --app-bg-1: #f3f6fb;
            --app-bg-2: #eef2f8;
            --panel-bg: #ffffff;
            --panel-border: #dbe3ef;
            --text-strong: #10233e;
            --text-muted: #5f738f;
            --accent-1: #0ea5a8;
            --accent-2: #2563eb;
            --accent-soft: #e0f2fe;
            --shadow-sm: 0 6px 18px rgba(16, 35, 62, 0.08);
            --shadow-md: 0 18px 48px rgba(16, 35, 62, 0.12);
            --radius-md: 14px;
            --radius-sm: 10px;
            --space-1: 8px;
            --space-2: 16px;
            --space-3: 24px;
        }

        html, body, [class*="css"] {
            font-family: "Segoe UI", "Aptos", "Inter", "Helvetica Neue", sans-serif;
            color: var(--text-strong);
        }

        .stApp {
            background:
                radial-gradient(1100px 380px at 10% -5%, rgba(14, 165, 168, 0.12), transparent 60%),
                radial-gradient(1100px 380px at 95% 0%, rgba(37, 99, 235, 0.1), transparent 55%),
                linear-gradient(180deg, var(--app-bg-1) 0%, var(--app-bg-2) 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1f36 0%, #1a3354 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #e2ebf8;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] .st-emotion-cache-10trblm {
            color: #ffffff;
            letter-spacing: 0.02em;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.76);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--panel-border);
        }

        .app-shell-header {
            position: sticky;
            top: 0.35rem;
            z-index: 20;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
            padding: 10px 14px;
            margin-bottom: var(--space-2);
            backdrop-filter: blur(8px);
        }

        .app-shell-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }

        .app-shell-title {
            color: #0f2748;
            font-weight: 700;
            letter-spacing: 0.01em;
            font-size: 1rem;
        }

        .app-shell-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-muted);
            font-size: 0.86rem;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid;
            font-weight: 600;
            font-size: 0.78rem;
            line-height: 1.2;
        }

        .status-chip.ok {
            color: #065f46;
            background: #ecfdf5;
            border-color: #86efac;
        }

        .status-chip.warn {
            color: #7f1d1d;
            background: #fff1f2;
            border-color: #fda4af;
        }

        .main-title {
            font-size: 2.65rem;
            font-weight: 800;
            background: linear-gradient(120deg, var(--accent-1) 0%, var(--accent-2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 1.3rem;
        }

        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-md);
            padding: 0.75rem 0.95rem;
            box-shadow: var(--shadow-sm);
        }

        div[data-testid="stMetric"] label {
            color: var(--text-muted);
            font-weight: 600;
        }

        .kpi-card {
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            border-radius: var(--radius-md);
            border: 1px solid var(--panel-border);
            padding: var(--space-2);
            text-align: left;
            box-shadow: var(--shadow-sm);
            transition: transform 0.24s ease, box-shadow 0.24s ease, border-color 0.24s ease;
            border-left: 4px solid rgba(37, 99, 235, 0.55);
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            border-color: rgba(14, 165, 168, 0.5);
            box-shadow: var(--shadow-md);
        }

        .kpi-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 5px;
        }

        .kpi-icon {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            background: var(--accent-soft);
            color: #0f4ca8;
            border: 1px solid #bee3f8;
        }

        .kpi-trend {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            border: 1px solid;
        }

        .kpi-trend.up {
            background: #ecfdf5;
            color: #166534;
            border-color: #86efac;
        }

        .kpi-trend.flat {
            background: #f8fafc;
            color: #334155;
            border-color: #cbd5e1;
        }

        .kpi-trend.down {
            background: #fff1f2;
            color: #9f1239;
            border-color: #fda4af;
        }

        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(120deg, #0f766e 0%, #1d4ed8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .kpi-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 6px;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(120deg, var(--accent-1) 0%, var(--accent-2) 100%);
            color: white;
            border: 0;
            border-radius: 11px;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(14, 165, 168, 0.24);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(37, 99, 235, 0.24);
        }

        .stButton > button:focus,
        .stDownloadButton > button:focus,
        .stTextInput input:focus,
        .stSelectbox [data-baseweb="select"]:focus-within,
        .stSlider [role="slider"]:focus {
            outline: 2px solid rgba(37, 99, 235, 0.5) !important;
            outline-offset: 1px;
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox [data-baseweb="select"],
        .stMultiSelect [data-baseweb="select"] {
            border-radius: var(--radius-sm) !important;
            border-color: #cfd9e8 !important;
        }

        .stDataFrame,
        div[data-testid="stTable"] {
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            background: #ffffff;
        }

        .stDataFrame [role="columnheader"],
        div[data-testid="stTable"] thead tr th {
            position: sticky;
            top: 0;
            background: #f5f9ff !important;
            color: #1e3a5f !important;
            z-index: 2;
            border-bottom: 1px solid #d6e1f0 !important;
        }

        .stDataFrame [role="row"]:nth-child(even) {
            background: #fbfdff;
        }

        .stDataFrame [role="row"]:hover {
            background: #eef6ff;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-md);
            background: #ffffff;
            box-shadow: var(--shadow-sm);
        }

        .stPlotlyChart {
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-md);
            background: #ffffff;
            box-shadow: var(--shadow-sm);
            padding: 8px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 3px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            background: #f4f8ff;
            border: 1px solid #d9e5f5;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, rgba(14, 165, 168, 0.12), rgba(37, 99, 235, 0.12));
            border-color: rgba(37, 99, 235, 0.35);
        }

        .sev-critical {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.16) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-left: 5px solid #ef4444;
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 8px;
            color: #7f1d1d;
        }
        .sev-warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.16) 100%);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-left: 5px solid #f59e0b;
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 8px;
            color: #78350f;
        }
        .sev-info {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(37, 99, 235, 0.14) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-left: 5px solid #3b82f6;
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 8px;
            color: #1e3a8a;
        }
        .terminal-console {
            background-color: #0f172a;
            color: #38bdf8;
            font-family: monospace;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #1e293b;
            max-height: 250px;
            overflow-y: auto;
        }

        section[data-testid="stSidebar"] [data-baseweb="radio"] label {
            border-radius: 10px;
            padding: 7px 8px;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }

        section[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.16);
        }

        section[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
            font-weight: 700;
        }

        .stAlert {
            border-radius: var(--radius-md) !important;
            border: 1px solid #d8e4f3 !important;
            box-shadow: var(--shadow-sm);
        }

        @media (max-width: 900px) {
            .app-shell-header {
                top: 0.2rem;
                padding: 9px 11px;
            }
            .app-shell-title {
                font-size: 0.95rem;
            }
            .main-title {
                font-size: 2rem;
            }
            .kpi-card {
                padding: 14px;
            }
            .kpi-value {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

