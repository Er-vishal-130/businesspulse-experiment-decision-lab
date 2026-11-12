# Kaggle A/B Testing Dataset

- **Dataset:** [A/B testing](https://www.kaggle.com/datasets/zhangluyuan/ab-testing)
- **Kaggle owner:** `zhangluyuan` (Luyuan Zhang)
- **Bundled file:** `ab_data.csv`
- **Downloaded version:** Kaggle dataset version 1
- **Download method:** `kagglehub.dataset_download("zhangluyuan/ab-testing")`

The original CSV is not included in the public repository because the Kaggle
license is listed as unknown. If you have verified redistribution rights, place
the file here as `ab_data.csv`. The application loader maps `timestamp` to
`date` and `landing_page` to `page_type` for its analysis interface without
changing the source file.

## Data-quality notes

The bundled CSV contains 294,478 rows and has no missing values in the source
columns. It contains repeated user IDs, so the platform suppresses inferential
analyses that require independent rows. It also contains records where `group`
does not match the corresponding expected landing page. These records are
preserved and reported in the app rather than silently corrected or removed.
Treat results as exploratory; the dataset is not verified evidence of a valid
randomized experiment.

## License and reuse

Kaggle currently reports the dataset license as **Unknown**. This attribution
does not grant additional rights. Check the dataset's current terms before
redistributing the CSV or including it in a public deployment.
