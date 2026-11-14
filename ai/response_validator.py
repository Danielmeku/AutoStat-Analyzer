"""Validate references and keep provider-authored prose separate from computed numbers."""
import re


def validate_response(response, context):
    if not isinstance(response, dict) or not isinstance(response.get("answer"), str):
        raise ValueError("The assistant returned an invalid response. Try again or use the offline guide.")
    answer, references = response['answer'].strip(), response.get('evidence_ids')
    if not answer or len(answer) > 3000 or not isinstance(references, list) or len(references) > 8:
        raise ValueError("The assistant response was outside the supported format.")
    allowed = {card['id'] for card in context['evidence']}
    if any(not isinstance(ref, str) or ref not in allowed for ref in references):
        raise ValueError("The assistant cited evidence that is not in the current analysis.")
    if re.search(r"\d", answer):
        raise ValueError("The assistant inserted numbers into its commentary. Use the computed evidence cards instead; try again or use the offline guide.")
    if re.search(r"\b(causes|proves|guarantees)\b", answer, re.I):
        raise ValueError("The assistant used unsupported certainty or causal language. Try again or use the offline guide.")
    return {"answer": answer, "evidence_ids": list(dict.fromkeys(references))}
