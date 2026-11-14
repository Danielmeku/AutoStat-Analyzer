from ai.response_validator import validate_response


def ask_assistant(provider, question, context):
    if not isinstance(question, str) or not question.strip() or len(question) > 1000:
        raise ValueError("Enter a question of 1–1,000 characters.")
    return validate_response(provider.respond(question.strip(), context), context)


def insight_record(provider, provider_name, context):
    response = ask_assistant(provider, "Explain the important findings, assumptions, limitations and next analytical steps in the supplied evidence.", context)
    if not response['evidence_ids']:
        raise ValueError('The assistant did not cite computed evidence for its insights. Try again or use the offline guide.')
    selected = {card['id'] for card in context['evidence']}
    return {"provider": provider_name, "commentary": response['answer'],
            "evidence": [card for card in context['evidence'] if card['id'] in response['evidence_ids'] and card['id'] in selected],
            "validation": "Evidence references checked. Numerical facts below are rendered from computed results, not provider prose. Commentary still requires human review; this is not a guarantee of factual or methodological correctness."}
