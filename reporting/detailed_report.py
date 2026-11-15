import io
import zipfile
from modules.data_profiler import profile_data
from modules.data_types import column_type_schema
from modules.exploratory_analysis import numeric_summary, outlier_summary, correlation_matrix
from reporting.executive_report import executive_report


def analysis_bundle(frame, dataset_name: str, analyses=None, pdfs=None) -> bytes:
    from reporting.results import json_result
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in ("executive.pdf", "detailed.pdf"):
            if filename in (pdfs or {}):
                archive.writestr(filename, pdfs[filename])
        summary = executive_report(frame, dataset_name)
        if analyses:
            summary += "\n\n## Additional analyses\nSee the included JSON results for the last completed test/model and their exact variable selections, exclusions, and limitations.\n"
        archive.writestr("summary.md", summary)
        for key, result in (analyses or {}).items():
            archive.writestr(f"{key}.json", json_result(result))
        archive.writestr("column_types.json", json_result(column_type_schema(frame)))
        archive.writestr("column_profile.csv", profile_data(frame).to_csv(index=False))
        archive.writestr("numeric_summary.csv", numeric_summary(frame).to_csv())
        archive.writestr("iqr_flags.csv", outlier_summary(frame).to_csv(index=False))
        archive.writestr("pearson_correlations.csv", correlation_matrix(frame).to_csv())
    return output.getvalue()
