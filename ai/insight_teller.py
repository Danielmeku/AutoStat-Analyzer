"""Deterministic observations from computed quality results, not an LLM."""

def quality_insights(quality: dict) -> list[str]:
    return [
        f"The active dataset contains {quality['rows']:,} rows and {quality['columns']} columns.",
        f"Missing cells account for {quality['missing_percent']:.2f}% of all cells ({quality['missing_cells']:,} cells).",
        f"There are {quality['duplicate_rows']:,} duplicate rows after the first occurrence. Confirm whether repeated records are meaningful before removing them.",
        "Exploratory associations do not establish causation. IQR flags are review candidates, not automatic errors.",
    ]
