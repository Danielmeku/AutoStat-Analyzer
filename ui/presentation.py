"""Dashboard components and the shared chart palette."""
import plotly.express as px
import plotly.io as pio
import streamlit as st

PALETTE = ['#B06CFF', '#24D9ED', '#F36DAE', '#688CFF', '#43D9B2']


def apply_style():
    template = pio.templates['plotly_dark'].to_plotly_json()
    template['layout'].update(
        paper_bgcolor='#151A2D', plot_bgcolor='#151A2D',
        font=dict(color='#DCE1F4', family='Arial', size=12), colorway=PALETTE,
        height=320, margin=dict(l=24, r=20, t=52, b=34),
        xaxis=dict(gridcolor='#242B43', zerolinecolor='#303956', showline=False),
        yaxis=dict(gridcolor='#242B43', zerolinecolor='#303956', showline=False),
        title=dict(font=dict(size=15, color='#F0F2FC'), x=.04),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    )
    pio.templates['autostat'] = template
    pio.templates.default = 'autostat'
    px.defaults.template = 'autostat'
    px.defaults.color_discrete_sequence = PALETTE
    st.markdown('''<style>
    .stApp {background:radial-gradient(ellipse at 85% 0%,#25213F 0%,#101425 46%,#0D1120 100%); color:#E8EAF6}
    [data-testid="stHeader"] {background:transparent}
    [data-testid="stSidebar"] {background:#0C1020; border-right:1px solid #252C44}
    [data-testid="stSidebar"] h2 {font-size:1.15rem; letter-spacing:-.03em}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:5px}
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding:7px 10px; border-radius:8px; transition:background .15s ease}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background:linear-gradient(90deg,#52317680,#29234855); box-shadow:inset 3px 0 #C084FC}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#20263D}
    .st-key-overview {margin:4px 0 14px}
    .st-key-overview [data-testid="stMetric"] {
        min-height:132px; border-radius:14px; padding:20px 22px;
        border:1px solid #FFFFFF18; box-shadow:0 8px 24px #00000024}
    .st-key-overview [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] {
        background:linear-gradient(115deg,#9561EE,#BA70E8); box-shadow:0 8px 28px #9856E02B}
    .st-key-overview [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {
        background:linear-gradient(115deg,#148FAF,#29CDD7); box-shadow:0 8px 28px #20BBD325}
    .st-key-overview [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {
        background:linear-gradient(115deg,#20233D,#2C2444); border-top:2px solid #EC78B4}
    .st-key-overview [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {
        background:linear-gradient(115deg,#1C293B,#1B223B); border-top:2px solid #7096FF}
    .st-key-overview [data-testid="stMetricLabel"] {color:#F4F1FF; font-size:.82rem}
    .st-key-overview [data-testid="stMetricValue"] {color:#FFFFFF; font-size:2rem; font-weight:700}
    [data-testid="stMetric"] {background:#1A2036; border:1px solid #2C3450; border-radius:12px; padding:16px}
    [data-testid="stPlotlyChart"] {background:#151A2D; border:1px solid #2A314A;
        border-radius:14px; padding:5px; overflow:hidden; box-shadow:0 6px 20px #0000001C}
    [data-testid="stDataFrame"] {border:1px solid #2A314A; border-radius:12px; overflow:hidden}
    [data-testid="stExpander"] {background:#151A2D; border-radius:10px; border-color:#2A314A}
    .block-container {padding-top:2.3rem; padding-bottom:3rem; max-width:1550px}
    h1 {letter-spacing:-.045em; font-size:2rem!important} h2,h3 {letter-spacing:-.025em}
    [data-testid="stCaptionContainer"] {color:#A8B1CE}
    .stButton > button[kind="primary"] {
        background:linear-gradient(110deg,#8B5CF6,#AD63E9); border:0; color:white;
        border-radius:9px; box-shadow:0 4px 15px #8B5CF630}
    .stButton > button[kind="primary"]:hover {filter:brightness(1.12); color:white}
    .workspace-heading {display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:22px}
    .workspace-heading h1 {padding:0;margin:0;font-size:1.8rem!important}
    .workspace-eyebrow {color:#AAB3D3;font-size:.68rem;letter-spacing:.18em;margin-bottom:6px}
    .workspace-subtitle {color:#929DBD;font-size:.85rem;margin-top:6px}
    .workspace-badge {color:#76E5DF;background:#18343D;border:1px solid #2E626A;border-radius:20px;padding:7px 12px;font-size:.72rem;white-space:nowrap}
    @media (max-width:640px) {
        .workspace-heading {align-items:flex-start}.workspace-badge {display:none}
        .block-container {padding-top:1.5rem}.st-key-overview [data-testid="stMetric"] {min-height:100px}
    }
    </style>''', unsafe_allow_html=True)


def workspace_header():
    st.markdown('''<div class="workspace-heading"><div>
    <div class="workspace-eyebrow">ANALYTICS WORKSPACE</div>
    <h1>AutoStat Analyzer</h1><div class="workspace-subtitle">Explore patterns. Understand results. Share insights.</div>
    </div><div class="workspace-badge">● &nbsp; Session workspace</div></div>''', unsafe_allow_html=True)


def plot_chart(figure, **kwargs):
    """Keep Streamlit's default chart theme from replacing dashboard colours."""
    figure.update_layout(
        template='autostat', height=320,
        margin=dict(l=60, r=24, t=52, b=55),
        xaxis=dict(automargin=True), yaxis=dict(automargin=True),
    )
    st.plotly_chart(figure, theme=None, **kwargs)


def result_details(value):
    with st.expander('View result JSON'):
        st.json(value)
