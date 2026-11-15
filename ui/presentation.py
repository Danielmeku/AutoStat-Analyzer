"""Shared dashboard styling and optional technical details."""
import plotly.express as px
import plotly.io as pio
import streamlit as st


def apply_style():
    template = pio.templates['plotly_dark'].to_plotly_json()
    template['layout'].update(
        paper_bgcolor='#14182D', plot_bgcolor='#14182D',
        font=dict(color='#E8EAF6'), colorway=['#A78BFA', '#22D3EE', '#F472B6', '#60A5FA'],
        height=360, margin=dict(l=30, r=20, t=55, b=35),
    )
    pio.templates['autostat'] = template
    pio.templates.default = 'autostat'
    px.defaults.template = 'autostat'
    px.defaults.color_discrete_sequence = ['#A78BFA', '#22D3EE', '#F472B6', '#60A5FA']
    st.markdown('''<style>
    .stApp {background:#0D1020; color:#E8EAF6}
    [data-testid="stSidebar"] {background:#111429; border-right:1px solid #282D48}
    [data-testid="stMetric"] {background:linear-gradient(135deg,#23213F,#171D36);
        border:1px solid #353351; border-radius:14px; padding:18px}
    [data-testid="stMetricValue"] {color:#C4B5FD}
    [data-testid="stPlotlyChart"] {border:1px solid #282D48; border-radius:14px; overflow:hidden}
    .block-container {padding-top:2rem; padding-bottom:3rem; max-width:1600px}
    h1 {letter-spacing:-.045em} h2,h3 {letter-spacing:-.025em}
    .stButton > button[kind="primary"] {background:#8B5CF6; border-color:#8B5CF6; color:white}
    </style>''', unsafe_allow_html=True)


def result_details(value):
    with st.expander('View result JSON'):
        st.json(value)
