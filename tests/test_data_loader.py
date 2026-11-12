import hashlib

import pandas as pd

from modules.data_loader import (
    KAGGLE_FILE,
    build_upload_metadata,
    load_kaggle_dataset,
)


def test_load_kaggle_dataset_uses_platform_column_names():
    assert KAGGLE_FILE.is_file()
    frame = load_kaggle_dataset()

    assert {"user_id", "date", "group", "page_type", "converted"} <= set(frame.columns)
    assert "timestamp" not in frame.columns
    assert "landing_page" not in frame.columns
    assert len(frame) == 294_478
    assert frame["user_id"].duplicated().any()


def test_upload_metadata_records_integrity_and_shape_without_file_contents():
    file_bytes = b"group,converted,device\nControl,0,Mobile\nTreatment,,Web\n"
    frame = pd.DataFrame(
        {
            "group": ["Control", "Treatment"],
            "converted": [0, None],
            "device": ["Mobile", "Web"],
        }
    )

    metadata = build_upload_metadata(
        "experiment.csv",
        file_bytes,
        frame,
        uploaded_at="2026-10-16T12:00:00+00:00",
    )

    assert metadata["filename"] == "experiment.csv"
    assert metadata["uploaded_at"] == "2026-10-16T12:00:00+00:00"
    assert metadata["row_count"] == 2
    assert metadata["column_count"] == 3
    assert metadata["file_size_bytes"] == len(file_bytes)
    assert metadata["sha256"] == hashlib.sha256(file_bytes).hexdigest()
    assert metadata["columns"]["converted"] == "float64"
    assert metadata["missing_values"] == {"converted": 1}
    assert "contents" not in metadata
