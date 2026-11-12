"""
Professional UI Theme System for BusinessPulse AI
Modern SaaS-inspired design with glassmorphism and smooth animations
"""

def get_premium_theme_css(dark_mode: bool = True) -> str:
    """
    Generate premium CSS theme for BusinessPulse AI.

    Design inspiration: Mixpanel, Amplitude, Modern SaaS dashboards
    Features: Glassmorphism, smooth transitions, professional spacing
    """

    if dark_mode:
        # Dark Mode Color Palette
        colors = {
            'bg_primary': '#0a0e1a',
            'bg_secondary': '#111827',
            'bg_tertiary': '#1a2332',
            'surface': 'rgba(17, 24, 39, 0.8)',
            'surface_hover': 'rgba(30, 41, 59, 0.9)',
            'border': 'rgba(99, 102, 241, 0.15)',
            'border_hover': 'rgba(99, 102, 241, 0.3)',
            'text_primary': '#f1f5f9',
            'text_secondary': '#cbd5e1',
            'text_muted': '#94a3b8',
            'accent_primary': '#6366f1',
            'accent_secondary': '#818cf8',
            'accent_gradient': 'linear-gradient(135deg, #6366f1 0%, #818cf8 100%)',
            'success': '#10b981',
            'success_bg': 'rgba(16, 185, 129, 0.1)',
            'warning': '#f59e0b',
            'warning_bg': 'rgba(245, 158, 11, 0.1)',
            'danger': '#ef4444',
            'danger_bg': 'rgba(239, 68, 68, 0.1)',
            'info': '#3b82f6',
            'info_bg': 'rgba(59, 130, 246, 0.1)',
            'shadow_sm': '0 2px 8px rgba(0, 0, 0, 0.3)',
            'shadow_md': '0 4px 16px rgba(0, 0, 0, 0.4)',
            'shadow_lg': '0 8px 32px rgba(0, 0, 0, 0.5)',
            'shadow_accent': '0 4px 24px rgba(99, 102, 241, 0.3)',
        }
    else:
        # Light Mode Color Palette
        colors = {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8fafc',
            'bg_tertiary': '#f1f5f9',
            'surface': 'rgba(255, 255, 255, 0.95)',
            'surface_hover': 'rgba(248, 250, 252, 1)',
            'border': 'rgba(99, 102, 241, 0.12)',
            'border_hover': 'rgba(99, 102, 241, 0.25)',
            'text_primary': '#0f172a',
            'text_secondary': '#475569',
            'text_muted': '#64748b',
            'accent_primary': '#4f46e5',
            'accent_secondary': '#6366f1',
            'accent_gradient': 'linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)',
            'success': '#059669',
            'success_bg': 'rgba(5, 150, 105, 0.08)',
            'warning': '#d97706',
            'warning_bg': 'rgba(217, 119, 6, 0.08)',
            'danger': '#dc2626',
            'danger_bg': 'rgba(220, 38, 38, 0.08)',
            'info': '#2563eb',
            'info_bg': 'rgba(37, 99, 235, 0.08)',
            'shadow_sm': '0 1px 3px rgba(0, 0, 0, 0.08)',
            'shadow_md': '0 4px 12px rgba(0, 0, 0, 0.1)',
            'shadow_lg': '0 10px 40px rgba(0, 0, 0, 0.15)',
            'shadow_accent': '0 4px 20px rgba(79, 70, 229, 0.25)',
        }

    return f"""
    <style>
    /* ============================================
       BASE STYLES & TYPOGRAPHY
       ============================================ */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    .stApp {{
        background: {colors['bg_primary']} !important;
        color: {colors['text_primary']} !important;
    }}

    /* ============================================
       CONTAINER & LAYOUT
       ============================================ */

    .block-container {{
        padding: 2rem 2.5rem 3rem !important;
        max-width: 1400px !important;
    }}

    /* ============================================
       SIDEBAR ENHANCEMENT
       ============================================ */

    div[data-testid="stSidebar"] {{
        background: {colors['bg_secondary']} !important;
        border-right: 1px solid {colors['border']} !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.1);
    }}

    div[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        color: {colors['text_secondary']} !important;
        cursor: pointer !important;
    }}

    div[data-testid="stSidebar"] .stRadio label:hover {{
        background: {colors['surface_hover']} !important;
        color: {colors['text_primary']} !important;
        transform: translateX(4px);
    }}

    /* ============================================
       PREMIUM METRIC CARDS
       ============================================ */

    div[data-testid="metric-container"] {{
        background: {colors['surface']} !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 16px !important;
        padding: 1.5rem 1.75rem !important;
        box-shadow: {colors['shadow_md']} !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    div[data-testid="metric-container"]:hover {{
        transform: translateY(-4px) !important;
        box-shadow: {colors['shadow_accent']} !important;
        border-color: {colors['border_hover']} !important;
    }}

    div[data-testid="metric-container"] label {{
        color: {colors['text_muted']} !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-bottom: 0.5rem !important;
    }}

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-size: 2rem !important;
        font-weight: 800 !important;
        background: {colors['accent_gradient']} !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }}

    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
        font-size: 0.875rem !important;
        font-weight: 600 !important;
    }}

    /* ============================================
       MODERN TABS
       ============================================ */

    .stTabs [data-baseweb="tab-list"] {{
        background: {colors['surface']} !important;
        border-radius: 14px !important;
        border: 1px solid {colors['border']} !important;
        padding: 6px !important;
        gap: 4px !important;
        box-shadow: {colors['shadow_sm']} !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {colors['text_secondary']} !important;
        border-radius: 10px !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        padding: 0.625rem 1.25rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: {colors['surface_hover']} !important;
        color: {colors['text_primary']} !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: {colors['accent_gradient']} !important;
        color: white !important;
        box-shadow: {colors['shadow_accent']} !important;
    }}

    /* ============================================
       PREMIUM BUTTONS
       ============================================ */

    .stButton > button {{
        background: {colors['accent_gradient']} !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: {colors['shadow_accent']} !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 28px rgba(99, 102, 241, 0.45) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* ============================================
       FORM INPUTS
       ============================================ */

    .stTextInput input, .stNumberInput input, .stSelectbox select {{
        background: {colors['surface']} !important;
        border: 1.5px solid {colors['border']} !important;
        border-radius: 12px !important;
        color: {colors['text_primary']} !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
    }}

    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {{
        border-color: {colors['accent_primary']} !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        outline: none !important;
    }}

    /* ============================================
       EXPANDERS
       ============================================ */

    details[data-testid="stExpander"] {{
        border: 1px solid {colors['border']} !important;
        border-radius: 14px !important;
        background: {colors['surface']} !important;
        margin-bottom: 1rem !important;
        box-shadow: {colors['shadow_sm']} !important;
        transition: all 0.2s ease !important;
    }}

    details[data-testid="stExpander"]:hover {{
        border-color: {colors['border_hover']} !important;
        box-shadow: {colors['shadow_md']} !important;
    }}

    details[data-testid="stExpander"] summary {{
        font-weight: 600 !important;
        color: {colors['text_primary']} !important;
        padding: 1rem 1.25rem !important;
        cursor: pointer !important;
        transition: color 0.2s ease !important;
    }}

    details[data-testid="stExpander"] summary:hover {{
        color: {colors['accent_primary']} !important;
    }}

    /* ============================================
       DATA TABLES
       ============================================ */

    [data-testid="stDataFrame"] {{
        border: 1px solid {colors['border']} !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: {colors['shadow_sm']} !important;
    }}

    [data-testid="stDataFrame"] table {{
        font-size: 0.875rem !important;
    }}

    [data-testid="stDataFrame"] th {{
        background: {colors['bg_tertiary']} !important;
        color: {colors['text_secondary']} !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
        padding: 1rem 0.75rem !important;
    }}

    [data-testid="stDataFrame"] td {{
        padding: 0.875rem 0.75rem !important;
        border-bottom: 1px solid {colors['border']} !important;
    }}

    /* ============================================
       ALERTS & NOTIFICATIONS
       ============================================ */

    div[data-testid="stAlert"] {{
        border-radius: 14px !important;
        border-left-width: 4px !important;
        font-size: 0.875rem !important;
        padding: 1rem 1.25rem !important;
        box-shadow: {colors['shadow_sm']} !important;
    }}

    /* ============================================
       CUSTOM COMPONENTS
       ============================================ */

    /* Hero Section */
    .bp-hero {{
        background: {colors['accent_gradient']};
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: {colors['shadow_accent']};
    }}

    .bp-hero::before {{
        content: "";
        position: absolute;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        pointer-events: none;
    }}

    .bp-hero h1 {{
        font-size: 2rem;
        font-weight: 900;
        color: white;
        margin: 0 0 0.5rem;
        letter-spacing: -0.02em;
    }}

    .bp-hero p {{
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0;
        font-weight: 500;
    }}

    /* KPI Cards */
    .bp-kpi-card {{
        background: {colors['surface']};
        backdrop-filter: blur(20px);
        border: 1px solid {colors['border']};
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        box-shadow: {colors['shadow_md']};
        transition: all 0.3s ease;
    }}

    .bp-kpi-card:hover {{
        transform: translateY(-4px);
        box-shadow: {colors['shadow_accent']};
    }}

    .bp-kpi-label {{
        color: {colors['text_muted']};
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
        display: block;
    }}

    .bp-kpi-value {{
        font-size: 2.25rem;
        font-weight: 900;
        background: {colors['accent_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }}

    .bp-kpi-change {{
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }}

    .bp-kpi-change.positive {{
        color: {colors['success']};
    }}

    .bp-kpi-change.negative {{
        color: {colors['danger']};
    }}

    /* Insight Cards */
    .bp-insight-card {{
        background: {colors['surface']};
        border-left: 4px solid {colors['accent_primary']};
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: {colors['shadow_md']};
        transition: all 0.2s ease;
    }}

    .bp-insight-card:hover {{
        transform: translateX(4px);
        box-shadow: {colors['shadow_lg']};
    }}

    .bp-insight-card.critical {{
        border-left-color: {colors['danger']};
    }}

    .bp-insight-card.high {{
        border-left-color: {colors['warning']};
    }}

    .bp-insight-card.medium {{
        border-left-color: {colors['info']};
    }}

    .bp-insight-card.low {{
        border-left-color: {colors['success']};
    }}

    .bp-insight-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }}

    .bp-insight-priority {{
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .bp-insight-priority.critical {{
        background: {colors['danger_bg']};
        color: {colors['danger']};
    }}

    .bp-insight-priority.high {{
        background: {colors['warning_bg']};
        color: {colors['warning']};
    }}

    .bp-insight-priority.medium {{
        background: {colors['info_bg']};
        color: {colors['info']};
    }}

    .bp-insight-priority.low {{
        background: {colors['success_bg']};
        color: {colors['success']};
    }}

    .bp-confidence-badge {{
        display: inline-flex;
        align-items: center;
        background: {colors['accent_gradient']};
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        box-shadow: {colors['shadow_accent']};
    }}

    /* Section Headers */
    .bp-section-header {{
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid {colors['border']};
    }}

    .bp-section-icon {{
        width: 40px;
        height: 40px;
        background: {colors['accent_gradient']};
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: {colors['shadow_accent']};
    }}

    .bp-section-header h2 {{
        font-size: 1.5rem;
        font-weight: 800;
        color: {colors['text_primary']};
        margin: 0;
        letter-spacing: -0.01em;
    }}

    /* Stats Pills */
    .bp-stat-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        color: {colors['text_primary']};
        margin: 0.25rem;
        transition: all 0.2s ease;
    }}

    .bp-stat-pill:hover {{
        border-color: {colors['accent_primary']};
        transform: translateY(-2px);
    }}

    .bp-stat-pill strong {{
        color: {colors['accent_primary']};
        font-weight: 700;
    }}

    /* Info Boxes */
    .bp-info-box {{
        background: {colors['info_bg']};
        border: 1px solid {colors['info']};
        border-left-width: 4px;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.875rem;
        color: {colors['text_primary']};
        margin: 1rem 0;
    }}

    .bp-warning-box {{
        background: {colors['warning_bg']};
        border: 1px solid {colors['warning']};
        border-left-width: 4px;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.875rem;
        color: {colors['text_primary']};
        margin: 1rem 0;
    }}

    .bp-success-box {{
        background: {colors['success_bg']};
        border: 1px solid {colors['success']};
        border-left-width: 4px;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 0.875rem;
        color: {colors['text_primary']};
        margin: 1rem 0;
    }}

    /* Timeline */
    .bp-timeline {{
        position: relative;
        padding-left: 2.5rem;
    }}

    .bp-timeline-step {{
        position: relative;
        padding: 1.5rem 0;
    }}

    .bp-timeline-step::before {{
        content: "";
        position: absolute;
        left: -2.5rem;
        top: 0;
        width: 16px;
        height: 16px;
        background: {colors['accent_primary']};
        border-radius: 50%;
        box-shadow: 0 0 0 4px {colors['surface']}, 0 0 0 6px {colors['accent_primary']};
    }}

    .bp-timeline-step::after {{
        content: "";
        position: absolute;
        left: -2.1rem;
        top: 16px;
        width: 2px;
        height: calc(100% + 1.5rem);
        background: {colors['border']};
    }}

    .bp-timeline-step:last-child::after {{
        display: none;
    }}

    /* Badge System */
    .bp-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .bp-badge-success {{
        background: {colors['success_bg']};
        color: {colors['success']};
        border: 1px solid {colors['success']};
    }}

    .bp-badge-warning {{
        background: {colors['warning_bg']};
        color: {colors['warning']};
        border: 1px solid {colors['warning']};
    }}

    .bp-badge-danger {{
        background: {colors['danger_bg']};
        color: {colors['danger']};
        border: 1px solid {colors['danger']};
    }}

    .bp-badge-info {{
        background: {colors['info_bg']};
        color: {colors['info']};
        border: 1px solid {colors['info']};
    }}

    .bp-badge-accent {{
        background: rgba(99, 102, 241, 0.15);
        color: {colors['accent_primary']};
        border: 1px solid {colors['accent_primary']};
    }}

    /* Loading States */
    .bp-loading {{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        font-size: 0.875rem;
        color: {colors['text_muted']};
    }}

    /* Empty States */
    .bp-empty-state {{
        text-align: center;
        padding: 4rem 2rem;
        color: {colors['text_muted']};
    }}

    .bp-empty-state-icon {{
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.3;
    }}

    .bp-empty-state h3 {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {colors['text_secondary']};
        margin-bottom: 0.5rem;
    }}

    .bp-empty-state p {{
        font-size: 0.875rem;
        color: {colors['text_muted']};
    }}

    /* ============================================
       SCROLLBAR
       ============================================ */

    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: transparent;
    }}

    ::-webkit-scrollbar-thumb {{
        background: rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        transition: background 0.2s ease;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(99, 102, 241, 0.5);
    }}

    /* ============================================
       RESPONSIVE ADJUSTMENTS
       ============================================ */

    @media (max-width: 768px) {{
        .block-container {{
            padding: 1rem !important;
        }}

        .bp-hero h1 {{
            font-size: 1.5rem;
        }}

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
            font-size: 1.5rem !important;
        }}
    }}

    /* ============================================
       ANIMATIONS
       ============================================ */

    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes slideIn {{
        from {{
            transform: translateX(-10px);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}

    .bp-animate-fade {{
        animation: fadeIn 0.4s ease-out;
    }}

    .bp-animate-slide {{
        animation: slideIn 0.3s ease-out;
    }}

    </style>
    """
