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
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
        }
        .kpi-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.4);
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .kpi-label {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 5px;
        }
        .sev-critical {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.2) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-left: 5px solid #ef4444;
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 8px;
            color: #fca5a5;
        }
        .sev-warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.2) 100%);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-left: 5px solid #f59e0b;
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 8px;
            color: #fde047;
        }
        .sev-info {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(37, 99, 235, 0.2) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-left: 5px solid #3b82f6;
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 8px;
            color: #93c5fd;
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
        </style>
        """,
        unsafe_allow_html=True,
    )

