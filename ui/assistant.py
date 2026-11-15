import hashlib
import os
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from ai.context_builder import build_context
from ai.chatbot import ask_assistant, insight_record
from ai.providers.offline_provider import OfflineProvider
from ai.providers.gemini_provider import GeminiProvider
from reporting.results import json_result


def _show_evidence(cards):
    for card in cards:
        st.markdown(f"**{card['id']} · {card['title']}**")
        st.text(card['statement'])


def render_assistant(frame, analyses, data_signature, digits):
    st.subheader("Statistical assistant & insights")
    context = build_context(frame, analyses, digits)
    signature = hashlib.sha256((data_signature + str(digits) + json_result(context)).encode()).hexdigest()
    if st.session_state.get("assistant_signature") != signature:
        for key in ["assistant_messages", "insight_result"]:
            st.session_state.pop(key, None)
        st.session_state['assistant_signature'] = signature
    selected = st.radio("Assistant mode", ["Offline guide", "Gemini"], horizontal=True)
    with st.expander("Review the summary available to the assistant", expanded=False):
        st.json(context)
        st.caption("No raw rows, category values, row predictions, PCA coordinates or earlier chat messages are sent. Column/target names, computed aggregates and your current question can still be sensitive.")
    configured = True
    api_key, model = os.environ.get('GEMINI_API_KEY'), os.environ.get('GEMINI_MODEL')
    if selected == 'Gemini':
        st.caption("Gemini actions send the summary and your question to Google.")
        if not (api_key and model):
            try:
                api_key = api_key or st.secrets.get('GEMINI_API_KEY')
                model = model or st.secrets.get('GEMINI_MODEL')
            except (FileNotFoundError, StreamlitSecretNotFoundError):
                pass
        configured = bool(api_key and model)
        if not configured:
            st.info("Live Gemini needs GEMINI_API_KEY and GEMINI_MODEL as environment variables or Streamlit secrets. Configure the key securely and restart; do not paste it into chat or source code. The offline guide works now.")
    def provider():
        return OfflineProvider() if selected == 'Offline guide' else GeminiProvider(api_key, model)
    if st.button("Generate insights", type='primary', disabled=not configured):
        try:
            with st.spinner("Preparing an evidence-based explanation…"):
                st.session_state['insight_result'] = insight_record(provider(), selected, context)
        except ValueError as exc:
            st.error(str(exc))
    if insight := st.session_state.get('insight_result'):
        st.markdown(f"#### Saved insights · {insight['provider']}")
        st.text(insight['commentary'])
        _show_evidence(insight['evidence'])
    st.markdown("#### Ask about your results")
    with st.form('assistant_question_form', clear_on_submit=True):
        question = st.text_input("Your statistical question", max_chars=1000)
        submitted = st.form_submit_button("Ask assistant", disabled=not configured)
    if submitted:
        try:
            with st.spinner("Preparing an explanation…"):
                response = ask_assistant(provider(), question, context)
            references = set(response['evidence_ids'])
            message = {"question": question, "provider": selected, "answer": response['answer'],
                       "evidence": [card for card in context['evidence'] if card['id'] in references]}
            st.session_state['assistant_messages'] = (st.session_state.get('assistant_messages', []) + [message])[-10:]
        except ValueError as exc:
            st.error(str(exc))
    for message in st.session_state.get('assistant_messages', []):
        with st.chat_message('user'):
            st.text(message['question'])
        with st.chat_message('assistant'):
            st.caption(message['provider'])
            st.text(message['answer'])
            _show_evidence(message['evidence'])
