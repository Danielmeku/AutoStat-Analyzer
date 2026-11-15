"""Column typing is opt-in and applied to a separate working copy."""
import streamlit as st
from modules.data_types import TYPE_OPTIONS, prepare_working_copy, preparation_description


def render_column_types(original):
    st.markdown('#### Column types')
    assignments = st.session_state.get('column_type_overrides', {})
    # Submit column and treatment together, so changing the column cannot reset
    # a pending choice while a separate widget rerun is still in flight.
    with st.form('column_type_form', border=False):
        columns = st.columns([1.4, 1.4, 1], vertical_alignment='bottom')
        column = columns[0].selectbox('Column to edit', list(original.columns))
        treatment = columns[1].selectbox('Treat as', TYPE_OPTIONS,
                                       key='column_type_choice_target',
                                       help='Factor values are categories, even when their labels are numbers. Date/time accepts ISO dates and normalizes to UTC. Conversions never silently discard invalid values.')
        submitted = columns[2].form_submit_button('Apply column type', type='primary', width='stretch')
    if submitted:
        updated = dict(assignments)
        if treatment == 'Original type':
            updated.pop(column, None)
        else:
            updated[column] = treatment
        cleaning = st.session_state.get('applied_cleaning', {'remove_duplicates': False, 'drop_missing': False})
        try:
            active = prepare_working_copy(original, updated, **cleaning)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state['column_type_overrides'] = updated
            st.session_state['active'] = active
            st.session_state['preparation'] = preparation_description(updated, **cleaning)
            st.rerun()
