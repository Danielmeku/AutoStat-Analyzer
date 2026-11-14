"""Optional Gemini REST provider; no key is logged, saved, or exposed in UI output."""
import json
import re
import requests
from ai.prompt_templates import SYSTEM_PROMPT


class GeminiProvider:
    def __init__(self, api_key, model):
        if not isinstance(api_key, str) or not isinstance(model, str) or not api_key or not model:
            raise ValueError("Set GEMINI_API_KEY and GEMINI_MODEL securely in your hosting environment, then restart the app.")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", model):
            raise ValueError("GEMINI_MODEL must be a model identifier, not a URL or path.")
        self._api_key, self.model = api_key, model

    def respond(self, question, context):
        body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": json.dumps({"question": question, "computed_evidence": context}, ensure_ascii=False)}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500, "responseMimeType": "application/json"}}
        try:
            response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                                     headers={"x-goog-api-key": self._api_key}, json=body,
                                     timeout=(5, 45), allow_redirects=False)
        except requests.RequestException:
            raise ValueError("Could not reach Gemini. Check network access and try again, or use the offline guide.") from None
        if response.status_code != 200:
            messages = {400: "Gemini rejected the request. Check model availability and configuration.",
                        401: "Gemini authentication failed. Check the configured API key.",
                        403: "Gemini access was denied. Check API permissions, regional access and network policy.",
                        404: "The configured Gemini model was not found. Set GEMINI_MODEL to an available model.",
                        429: "Gemini quota or rate limit reached. Try later or use the offline guide."}
            raise ValueError(messages.get(response.status_code, "Gemini did not complete the request. Try later or use the offline guide."))
        try:
            parts = response.json()['candidates'][0]['content']['parts']
            if not isinstance(parts, list) or any(not isinstance(part, dict) for part in parts):
                raise ValueError
            text = ''.join(part.get('text', '') for part in parts if not part.get('thought'))
            if len(text) > 12000:
                raise ValueError
            return json.loads(text)
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError("Gemini returned no usable structured explanation (possibly blocked or truncated). Try again or use the offline guide.") from None
