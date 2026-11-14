import io
import zipfile
from modules.data_profiler import profile_data
from modules.exploratory_analysis import numeric_summary, outlier_summary, correlation_matrix
from reporting.executive_report import executive_report


def analysis_bundle(frame, dataset_name: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.md", executive_report(frame, dataset_name))
        archive.writestr("column_profile.csv", profile_data(frame).to_csv(index=False))
        archive.writestr("numeric_summary.csv", numeric_summary(frame).to_csv())
        archive.writestr("iqr_flags.csv", outlier_summary(frame).to_csv(index=False))
        archive.writestr("pearson_correlations.csv", correlation_matrix(frame).to_csv())
    return output.getvalue()
