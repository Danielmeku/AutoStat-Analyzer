import json
import math

def json_result(result):
    def sanitize(value):
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return json.dumps(sanitize(result), indent=2, ensure_ascii=False, allow_nan=False)
