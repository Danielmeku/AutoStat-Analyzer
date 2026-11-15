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
        background:linear-gradient(115deg,#20233D,#2C2444); border-top:2px solid #B06CFF}
    .st-key-overview [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {
        background:linear-gradient(115deg,#1C293B,#1B223B); border-top:2px solid #24D9ED}
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
    .st-key-chat_launcher {position:fixed;right:24px;bottom:24px;width:60px;z-index:1000002}
    .st-key-chat_launcher button {width:60px;height:60px;border-radius:50%;padding:0;
        background:linear-gradient(135deg,#B652C5,#8B5CF6);color:white;border:2px solid #C692EC;
        box-shadow:0 6px 24px #0006;justify-content:center}
    .st-key-chat_launcher button [data-testid="stMarkdownContainer"],
    .st-key-assistant_panel .st-key-close_chat button [data-testid="stMarkdownContainer"] {
        position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}
    .st-key-chat_launcher button:focus-visible {outline:3px solid #24D9ED;outline-offset:4px}
    .st-key-assistant_panel {position:fixed;right:24px;bottom:98px;width:min(390px,calc(100vw - 32px));
        max-height:calc(100dvh - 126px);overflow-y:auto;overflow-x:hidden;z-index:1000002;
        background:#151A2D;border:1px solid #48506F!important;border-radius:18px!important;
        padding:16px!important;box-shadow:0 16px 60px #0009;box-sizing:border-box}
    .st-key-assistant_panel h3 {font-size:1.1rem;padding:0}
    .st-key-assistant_panel [data-testid="stVerticalBlock"] {gap:10px}
    .st-key-assistant_panel [data-testid="stHorizontalBlock"] {gap:8px;flex-wrap:nowrap}
    .st-key-assistant_panel [data-testid="stColumn"] {min-width:0!important}
    .st-key-assistant_panel .st-key-close_chat button {border:0;background:transparent;width:34px;height:34px;min-height:34px;padding:0}
    .st-key-chat_history:has([data-testid="stChatMessage"]) {height:min(240px,30dvh)!important;min-height:100px}
    .st-key-assistant_panel [data-testid="stChatMessage"] {padding:10px;border-radius:12px;background:#20263C}
    .st-key-assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {background:#563C79;width:fit-content;max-width:90%;margin-left:auto}
    .st-key-assistant_panel [data-testid="stChatMessageAvatarUser"] {display:none}
    .st-key-assistant_panel [data-testid="stChatMessageAvatarAssistant"] {background:#392D55;color:#CFA3FF}
    .st-key-assistant_panel [data-testid="stChatInput"] {border-color:#454665;background:#1C2239}
    .st-key-assistant_panel .stButton button {font-size:.8rem}
    .block-container {padding-bottom:6rem}
    @media (max-width:640px) {
        .st-key-chat_launcher {right:16px;bottom:16px}
        .st-key-assistant_panel {right:16px;bottom:88px;max-height:calc(100dvh - 160px)}
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
