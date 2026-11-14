import json
from ai.insight_teller import quality_insights
from modules.data_validator import validate_data


def executive_report(frame, dataset_name: str) -> str:
    quality = validate_data(frame)
    # JSON quoting keeps user-supplied names distinct from report formatting.
    lines = ["# AutoStat Analyzer — Exploratory summary", "", f"Dataset: {json.dumps(dataset_name)}", ""]
    lines.extend(f"- {line}" for line in quality_insights(quality))
    lines += ["", "## Scope", "Descriptive exploration only. No hypothesis tests or predictive models were run.",
              "Missing numeric values are excluded per column; correlations use pairwise complete observations (minimum 3).",
              "This report describes the active dataset, including any explicitly applied cleaning."]
    return "\n".join(lines)
