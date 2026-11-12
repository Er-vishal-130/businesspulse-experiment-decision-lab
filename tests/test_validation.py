import pandas as pd

from modules.validator import validate_dataframe


def test_valid_dataframe():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "experiment_id": ["exp_001", "exp_001", "exp_001"],
            "group": ["Control", "Treatment", "Control"],
            "converted": [0, 1, 0],
            "revenue": [0, 120, 0],
            "age": [25, 30, 35],
            "gender": ["Male", "Female", "Male"],
            "country": ["India", "India", "USA"],
            "device": ["Mobile", "Web", "Mobile"],
            "traffic_source": ["Organic", "Referral", "Organic"],
            "session_duration": [200, 350, 180],
            "pages_viewed": [4, 7, 6],
            "bounce": [0, 1, 0],
            "error_flag": [0, 0, 1],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    result = validate_dataframe(df)
    assert result["status"] == "Valid"


def test_invalid_dataframe_detects_issues():
    df = pd.DataFrame({"user_id": [1, 2], "group": ["Control", "Unknown"], "converted": [0, 2]})
    result = validate_dataframe(df)
    assert result["status"] in {"Warning", "Invalid"}
    assert len(result["issues"]) >= 1
