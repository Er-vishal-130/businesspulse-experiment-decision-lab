# BusinessPulse | Experiment Decision Lab

BusinessPulse is a Streamlit-based decision-support platform for A/B tests and business analysis. It allows users to upload or generate experimental data, validate quality, calculate conversion KPIs, perform statistical tests, compute confidence intervals and power, evaluate segment-level effects, analyze guardrails, and generate a recommendation that balances statistical evidence with business risk.

## Project Objective

The system supports the complete workflow:

DATA → VALIDATION → KPI → STATISTICAL ANALYSIS → UNCERTAINTY → SEGMENTATION → GUARDRAILS → BUSINESS IMPACT → DECISION

The platform is designed to help analysts understand not only whether a result is statistically significant, but also whether it is practically useful and operationally safe.

## Core Features

- CSV/Excel data upload and validation
- Optional Kaggle A/B testing dataset (not included in this public repository; its license is unknown)
- Built-in sample dataset generator
- Missing-value, duplicate, type, and group validation
- KPI dashboard for conversion, revenue, and lift metrics
- Two-proportion hypothesis testing (z-test and Fisher's exact)
- Confidence intervals and effect size analysis
- Power and minimum detectable effect calculation
- Segmentation analysis by device, country, age, traffic source, and more
- Guardrail metric checks
- Business impact estimate using monthly visitors and revenue assumptions
- BusinessPulse Detective anomaly workflow for observed-vs-expected comparisons
- Rule-based decision engine
- Report export and SQLite/PostgreSQL persistence
- Data-quality scoring and cleaning recommendations
- KPI intelligence, trends, period comparisons, and anomaly detection
- Evidence-based insights, root-cause drill-down, funnel, and cohort analytics

## New Features (v2.0)

- **Bayesian A/B Analysis** — Beta-Binomial posterior distributions, credible intervals, P(Treatment > Control), and expected loss
- **Multi-Metric A/B Testing** — Simultaneous testing of conversion, revenue, session duration, pages viewed, and bounce with Holm-Bonferroni family-wise error rate control
- **Sequential / Interim Analysis** — SPRT (likelihood ratio test) and O'Brien-Fleming alpha-spending boundary for early stopping decisions
- **Experiment History & Registry** — View and compare past experiments from the database; import external CSV history files; timeline chart across runs
- **Advanced Visualizations** — Conversion funnel (Control vs Treatment), Segment × Metric Z-score heatmap, Waterfall chart of conversion contributors, and daily trend ribbon
- **Enhanced Ask Your Data (20+ patterns)** — Revenue per user, total revenue, revenue per conversion, bounce rate, session duration, weekday/weekly trends, Top-N segment queries, and more
- **BusinessPulse AI workspace** — Dedicated tabs for automated insights, KPI monitoring and targets, root-cause investigations, funnel and cohort analysis, multi-method segment anomaly detection, and time-series anomaly screening
## Architecture Summary

The project follows a modular architecture:

- Streamlit UI layer
- Data ingestion and validation layer
- Business metric and statistical engine
- Segmentation and guardrail analysis
- Decision logic and report generation
- SQLite by default, with PostgreSQL support through `DATABASE_URL`

## Tech Stack

- Python 3.11+
- Streamlit
- Pandas, NumPy
- SciPy, Statsmodels
- Plotly
- SQLite / PostgreSQL
- Pytest

## Folder Structure

```text
ab_testing_platform/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_ab_test.csv
├── database/
│   ├── database.py
│   └── schema.sql
├── modules/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── validator.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── hypothesis_testing.py
│   ├── confidence_intervals.py
│   ├── effect_size.py
│   ├── power_analysis.py
│   ├── segmentation.py
│   ├── guardrails.py
│   ├── bayesian.py
│   ├── detective.py
│   ├── experiment_history.py
│   ├── multi_metric.py
│   ├── sequential.py
│   ├── decision_engine.py
│   └── report_generator.py
├── visualizations/
│   ├── charts.py
│   └── advanced_charts.py
├── tests/
│   ├── test_database.py
│   ├── test_data_loader.py
│   ├── test_business_analytics.py
│   ├── test_detective.py
│   ├── test_metrics.py
│   ├── test_power_analysis.py
│   ├── test_statistics.py
│   └── test_validation.py
├── docs/
│   ├── architecture.md
│   └── methodology.md
└── database/ab_testing.db (created automatically for SQLite)
```

`data/ab_data.csv` is intentionally omitted from the public repository because
its upstream license is unknown. See `data/KAGGLE_DATASET.md` before providing
that dataset locally.

## Quick Start

```bash
cd ab_testing_platform
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app starts with a generated synthetic dataset and does not require the Kaggle
CSV. The public repository intentionally excludes `data/ab_data.csv` because its
Kaggle license is listed as unknown. The Kaggle data-source option is only shown
when that file is provided locally and should not be enabled in a public
deployment unless you have verified redistribution rights.

## BusinessPulse Detective Demo

To showcase the anomaly workflow quickly, open the sidebar and choose the synthetic demo dataset. Then navigate to the BusinessPulse Detective page to see a baseline comparison, observed-vs-expected values, and the strongest anomaly signal. This demo is intentionally synthetic and clearly labeled as demo-only.

## PostgreSQL Configuration

SQLite is used automatically unless the `DATABASE_URL` environment variable is set. To use PostgreSQL, install the project requirements, create an empty database, set its connection URL, and launch the app:

```powershell
$env:DATABASE_URL = "postgresql://USERNAME:PASSWORD@HOST:5432/DATABASE_NAME"
python -m streamlit run app.py
```

The application creates its required tables automatically on startup. Keep the connection URL private; do not commit real credentials to source control. When `DATABASE_URL` is unset, the project continues to use `database/ab_testing.db`.

## Uploading Data

If you have verified redistribution rights and provide `data/ab_data.csv`
locally, the sidebar will include **Kaggle A/B dataset (zhangluyuan)**. The
loader maps `timestamp` to `date` and `landing_page` to `page_type`; the source
CSV itself is not modified. The source is
[Luyuan Zhang's A/B testing dataset on Kaggle](https://www.kaggle.com/datasets/zhangluyuan/ab-testing).
Kaggle lists its license as **Unknown**, so this repository and its public demo
do not redistribute the CSV. See
[`data/KAGGLE_DATASET.md`](data/KAGGLE_DATASET.md) for attribution and data notes.

This educational dataset contains repeated user IDs and rows where the assigned
group and landing page do not match. The application preserves these observations
and displays a warning; repeated IDs suppress analyses that require independent
rows. Do not treat its results as verified causal evidence.

Choose **Upload file** in the sidebar and select a `.csv`, `.xlsx`, or `.xls` file. A newly selected file replaces the active analysis dataset; the app keeps that upload selected while you navigate the dashboard.

For A/B analysis, only `group` and `converted` are required. `group` should contain `Control` and `Treatment`; `converted` must contain binary `0`/`1` outcomes. Other columns are optional and only enable their corresponding KPIs, filters, charts, and segmentation views: `user_id`, `session_id`, `experiment_id`, `revenue`, `age`, `gender`, `country`, `device`, `traffic_source`, `session_duration`, `pages_viewed`, `bounce`, `error_flag`, and `date`. Users are counted distinctly when `user_id` is present and sessions when `session_id` is present. Inference assumes independent randomized units; repeated `user_id` or `session_id` values suppress the independent-row A/B test, while unique or absent identifiers do not prove independence.

The **Ask Your Data** page supports rule-based questions about device/source conversion, the Control-versus-Treatment conversion result, segment revenue, and data-backed findings. Answers are calculated from the selected dataset and filters; unsupported questions or missing columns are identified rather than guessed.

The **BusinessPulse Detective** workflow compares Treatment segments against the Control reference population when both groups are available. Otherwise, each segment is compared with the remaining rows (leave-one-segment-out reference). It reports observed and expected conversions, absolute and relative changes, the test method, a Newcombe-Wilson 95% difference interval, and Bonferroni-adjusted p-values across tested segments. The UI minimum (default 30) is a configurable low-volume screening threshold, not evidence of adequate power; uncertainty and design limitations remain. Repeated `user_id` or `session_id` values suppress inferential alerts because the independent-row assumption is not verified. Exact duplicate rows are retained in descriptive calculations; when identity is repeated or uncertain, inferential alerts are suppressed rather than removing observations to force independence. For A/B comparisons, included segment excess-conversion contributions are reported separately from the full aggregate Treatment change; missing-category and low-volume rows are excluded with reasons, and the unreconciled residual is shown without redistribution. Segment comparisons are exploratory; adjusted significance does not establish causality.

When all available values are selected in the group/device/country/traffic-source filters, rows with missing values in those fields remain in the analysis so they can be counted and, where applicable, represented in excluded-category/residual accounting. Choosing only a subset of values filters out the unselected values and missing entries.

The **Alert Center** converts only the existing detector's adjusted-significant segment findings into alerts; it does not run a second detection method. Severity is assigned from editable absolute conversion-rate-difference bands (default 2, 5, and 10 percentage points for Medium, High, and Critical); these are operational settings, not validated universal thresholds. Alerts show their severity rule and evidence. Status changes (New, Investigating, Resolved) are stored in the existing SQLite/PostgreSQL database. The **KPI Drill-Down** reuses the Detective baseline and rows, ranks observed changes, and reports aggregate contribution/residual only where the Control/Treatment accounting supports it. Report exports include detector-supported alerts and data provenance. The detector's Bonferroni correction applies within a selected segment dimension; exploring multiple dimensions is not jointly corrected. All segment findings remain exploratory and non-causal.

Data source labels distinguish user-uploaded files, bundled sample data, generated samples, and the synthetic anomaly demo. For uploads, the session and exported reports include filename, timestamp, shape, inferred column types, missing-value counts, file size, and a SHA-256 file-byte fingerprint; raw uploads are not persisted as metadata. A fingerprint can help identify unchanged bytes, but does not establish file contents, origin, or business provenance. Uploading a file does not independently verify its business provenance. Synthetic data is clearly labeled and must never be presented as actual business performance.

The conversion comparison uses Fisher's exact test when the pooled expected cell counts are below five and a two-proportion z-test otherwise. Confidence intervals use Wilson intervals for individual rates and the Newcombe-Wilson method for the treatment-minus-control difference. Small groups and imbalanced assignments are flagged; repeated IDs suppress the independent-row test. No ID, or a set of unique IDs, is not proof of independence; repeated or cluster-randomized observations require a design-specific analysis.

The sample-size planner uses a normal approximation for an independent, two-group binary conversion test with equal allocation, specified target rates, alpha, desired power, and test sidedness. It returns unavailable for a zero target effect or boundary rates unsupported by that approximation. The dashboard MDE estimate uses the smaller valid arm size as a conservative equal-allocation approximation. These are design-planning calculations, not post-hoc evidence of achieved power; they do not establish that an uploaded experiment met the assumptions.

## Statistical Responsibility

The system does not claim that a low p-value alone means a treatment is valuable. The app explicitly considers:

- sample size
- confidence intervals
- effect size
- guardrail metrics
- business impact
- multiple comparison risk

A statistically significant result is still interpreted in the context of operational trade-offs.

## Public deployment on Streamlit Community Cloud

GitHub stores the source code; Streamlit Community Cloud runs the Python app.
After pushing this repository to GitHub:

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Select **Create app**, choose this repository and the `main` branch.
3. Set the main file path to `app.py` and deploy.
4. If using PostgreSQL, add `DATABASE_URL` in the app's **Settings → Secrets**.

The app can use its default SQLite database without configuration, but local
SQLite storage on hosted instances may not persist across restarts. Use a
managed PostgreSQL database for durable shared experiment history. Never put
database credentials in the repository. Public deployments are accessible to
anyone with the URL; do not upload private or personally identifiable data.

## License

This project is for educational and portfolio use.
