"""Gemini chat widget available from every analysis page."""
import hashlib
import os
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from ai.context_builder import build_context
from ai.chatbot import ask_assistant, insight_record
from ai.providers.gemini_provider import GeminiProvider
from reporting.results import json_result


def _toggle_chat():
    st.session_state['chat_open'] = not st.session_state.get('chat_open', False)


def _clear_chat():
    st.session_state.pop('assistant_messages', None)


def _credentials():
    api_key, model = os.environ.get('GEMINI_API_KEY'), os.environ.get('GEMINI_MODEL')
    if not (api_key and model):
        try:
            api_key = api_key or st.secrets.get('GEMINI_API_KEY')
            model = model or st.secrets.get('GEMINI_MODEL')
        except (FileNotFoundError, StreamlitSecretNotFoundError):
            pass
    return api_key, model


def _show_evidence(cards):
    for card in cards:
        st.markdown(f"**{card['title']}**")
        st.text(card['statement'])


def render_assistant(frame, analyses, data_signature, digits):
    with st.container(key='chat_launcher'):
        st.button('Chat with Gemini', icon=':material/chat_bubble:', key='toggle_chat',
                  help='Open or close the Gemini assistant', on_click=_toggle_chat)
    if not st.session_state.get('chat_open', False):
        return
    context = build_context(frame, analyses, digits) if frame is not None else {
        "purpose": "Explain statistical concepts; no dataset is currently loaded.",
        "limitations": "Do not invent dataset findings. Ask the user to load data and run an analysis for dataset-specific questions.",
        "evidence": [],
    }
    signature = hashlib.sha256((data_signature+str(digits)+json_result(context)).encode()).hexdigest()
    if st.session_state.get('assistant_signature') != signature:
        for key in ['assistant_messages', 'insight_result']:
            st.session_state.pop(key, None)
        st.session_state['assistant_signature'] = signature
    api_key, model = _credentials()
    configured = bool(api_key and model)
    with st.container(key='assistant_panel', border=True):
        title, close = st.columns([5, 1], vertical_alignment='center')
        title.markdown('### Gemini assistant')
        with close:
            st.button('Close chat', icon=':material/close:', key='close_chat', on_click=_toggle_chat)
        st.caption('Messages send your question and aggregate summary to Google.')
        if not configured:
            st.info('Chat is unavailable. Configure GEMINI_API_KEY and GEMINI_MODEL securely in your environment settings, then restart the app.')
        save, clear = st.columns(2)
        if save.button('Generate insights', disabled=not configured, width='stretch'):
            try:
                with st.spinner('Saving insights…'):
                    st.session_state['insight_result'] = insight_record(GeminiProvider(api_key, model), 'Gemini', context)
                st.session_state['assistant_notice'] = 'Insights saved to your analysis exports.'
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        clear.button('Clear chat', on_click=_clear_chat, width='stretch')
        if notice := st.session_state.pop('assistant_notice', None):
            st.success(notice)
        with st.expander('Review shared summary'):
            st.json(context)
        if insight := st.session_state.get('insight_result'):
            with st.expander('Saved insights'):
                st.text(insight['commentary'])
                _show_evidence(insight['evidence'])
        history_options = {'height': 240} if configured or st.session_state.get('assistant_messages') else {}
        history = st.container(border=False, key='chat_history', **history_options)
        quick_question = None
        if configured and not st.session_state.get('assistant_messages'):
            left, right = st.columns(2)
            if left.button('Explain results', disabled=not configured, width='stretch'):
                quick_question = 'Explain the completed results and their limitations.'
            if right.button('Review assumptions', disabled=not configured, width='stretch'):
                quick_question = 'Which assumptions should I review for the completed analyses?'
        submitted = st.chat_input('Ask about your analysis…', key='assistant_question', max_chars=1000, disabled=not configured)
        error = None
        if question := (submitted or quick_question):
            try:
                with st.spinner('Thinking…'):
                    response = ask_assistant(GeminiProvider(api_key, model), question, context)
                references = set(response['evidence_ids'])
                message = {'question': question, 'provider': 'Gemini', 'answer': response['answer'],
                           'evidence': [card for card in context['evidence'] if card['id'] in references]}
                st.session_state['assistant_messages'] = (st.session_state.get('assistant_messages', [])+[message])[-10:]
            except ValueError as exc:
                error = str(exc)
        with history:
            if configured and not st.session_state.get('assistant_messages'):
                with st.chat_message('assistant', avatar=':material/auto_awesome:'):
                    st.write('Hi! Ask me about your results, tests, or next analysis steps.')
            for message in st.session_state.get('assistant_messages', []):
                with st.chat_message('user'):
                    st.text(message['question'])
                with st.chat_message('assistant', avatar=':material/auto_awesome:'):
                    st.text(message['answer'])
                    if message['evidence']:
                        with st.expander('Computed evidence'):
                            _show_evidence(message['evidence'])
            if error:
                st.error(error)
