# Interactive A/B Testing and Business Analytics Platform

## 1. Short Introduction

This project is a web-based analytics platform that helps users assess A/B experiments using statistical evidence, business metrics, and operational guardrails. It turns experimental data into transparent analysis and a cautious, evidence-based recommendation.

## 2. Brief Summary

The application accepts CSV or Excel data, or creates clearly labeled sample data. It validates the dataset, calculates conversion and business metrics, compares Control and Treatment groups, estimates uncertainty and effect size, and supports power planning, segmentation, anomaly screening, and guardrail checks. The results are visualized in an interactive Streamlit dashboard and can be summarized or exported. SQLite is the default database; PostgreSQL can be configured for persistent experiment and alert records.

The platform is a decision-support tool, not an automatic proof of causality. Statistical interpretation depends on data quality, experiment design, and the assumptions described in this report. Sample and generated data are illustrative, not verified business evidence.

## 3. Project Introduction

Digital product teams routinely test changes to websites, applications, marketing journeys, and other customer experiences. Comparing a Control group with a Treatment group can help teams estimate whether a change affects an outcome such as conversion. However, a difference in observed rates may be random variation, may be too small to matter commercially, or may be accompanied by harm to another important metric.

This project brings the principal steps of an A/B analysis into one interface. Its main outcome is conversion, while optional fields can support revenue, user/session, date, and segment analysis. It combines data checks and descriptive metrics with inferential statistics and business context so that users can inspect the evidence behind a recommendation.

### 3.1 Challenges Addressed

- Incomplete, malformed, duplicated, or inconsistent input data can undermine analysis.
- A raw percentage difference does not communicate uncertainty or statistical evidence.
- A statistically detectable result may still have little practical or financial value.
- Small samples and repeated observations can invalidate or weaken standard inference.
- Improvements in the primary metric can coincide with deterioration in revenue, errors, bounce rate, or other guardrails.
- Segment exploration can produce misleading apparent effects if multiple comparisons and low-volume groups are ignored.
- Users may mistake synthetic examples, uploaded files, or a dashboard recommendation for verified business findings.

The platform addresses these challenges with validation, confidence intervals, effect-size and power-planning views, guardrail-aware decision logic, segment screening, and explicit warnings about assumptions and data provenance.

### 3.2 Role of Technology

- **Python** provides the application and analytics logic.
- **Streamlit** presents the interactive dashboard and user controls.
- **Pandas and NumPy** support tabular data handling and numerical calculations.
- **SciPy and Statsmodels** provide statistical procedures used by the analysis modules.
- **Plotly** provides interactive visualizations.
- **SQLite** stores local experiment metadata, results, runs, and alert statuses; **PostgreSQL** is an optional configured backend.
- **Pytest** supports automated verification of analytical and data-management behavior.

Technology makes the workflow repeatable and accessible, but it cannot correct a biased experiment design or verify the origin of user-uploaded business data.

### 3.3 Literature Survey

1. Kohavi et al. describe large-scale online controlled experiments and the engineering and analytical practices needed to run them reliably. Their work motivates treating experimentation as a complete process—not merely comparing two dashboard percentages—and informs this project's structured analysis and decision workflow [1].
2. Kohavi, Tang, and Xu provide a practical treatment of online controlled experiments, including experiment design, metric selection, trustworthy analysis, and interpretation. This supports the project's emphasis on assumptions, uncertainty, and business context [2].
3. Newcombe compares methods for interval estimation of the difference between independent proportions. The project uses a Newcombe-Wilson interval for the treatment-minus-control difference, alongside Wilson intervals for individual rates, so users can assess uncertainty around conversion changes [3].
4. Standard statistical guidance on hypothesis tests and power analysis motivates distinguishing evidence against a null hypothesis from practical importance and planning an experiment's sample size in advance. The project's power and minimum-detectable-effect calculations are planning approximations, not retrospective proof of achieved power.

### 3.4 Case Studies

The following are **illustrative scenarios for explaining how the implemented workflow can be used**. They are not results from a real company or a production deployment.

**Case study A — Checkout-page experiment:** A product team uploads records containing Control/Treatment assignment and binary conversion outcomes. The platform checks required fields and valid outcomes, compares conversion rates, shows a confidence interval and test result, and checks available guardrails such as revenue or error rate. The decision should remain inconclusive if the evidence is weak or a guardrail is impaired.

**Case study B — Segment anomaly screening:** An analyst uses the BusinessPulse Detective workflow to inspect conversion differences across available dimensions such as device or traffic source. When both experiment groups exist, Treatment segments are compared with the Control reference; otherwise, the detector uses a leave-one-segment-out reference. Low-volume screening, adjusted p-values, and repeated-identifier safeguards help contextualize alerts. Segment results are exploratory and do not by themselves establish causality.

## 4. Problem Statement

Teams may rely on spreadsheets or static dashboards that show conversion rates but do not jointly address data quality, statistical uncertainty, sample-size adequacy, business impact, and guardrail risk. This creates a risk of adopting ineffective or harmful product changes based on noisy or incomplete evidence.

The problem addressed by this project is the absence of an accessible, integrated workflow for validating experimental data, analyzing Control-versus-Treatment outcomes, communicating assumptions and uncertainty, and translating results into a cautious business recommendation.

## 5. Proposed Solution

The proposed system is a modular Streamlit application that implements the following workflow:

**Data source → validation → cleaning and filtering → KPI analysis → statistical analysis → uncertainty and effect size → segmentation and guardrails → business-impact assessment → recommendation and report**

Users can upload CSV/XLS/XLSX files or use labeled sample data. For A/B conversion analysis, `group` and binary `converted` outcomes are required. The application offers conversion and lift measures, statistical testing, confidence intervals, effect sizes, experiment-planning estimates, segment analysis, anomaly screening, guardrail evaluation, business-impact estimates, and report generation. The decision engine considers evidence and business context rather than treating a p-value alone as a success criterion.

## 6. Methodologies Used

### 6.1 Data and Validation Method

The input is checked for the required group and conversion fields, valid Control/Treatment groups, binary outcomes, missing values, duplicates, and other data-quality concerns. Optional columns are used only for corresponding analyses when present; unavailable business metrics are not invented or imputed as findings.

### 6.2 Descriptive and Business Metrics

- Conversion rate = conversions / eligible observations.
- Absolute lift = Treatment conversion rate − Control conversion rate.
- Relative lift = (Treatment rate − Control rate) / Control rate × 100%, when the Control rate permits calculation.
- Optional business metrics and estimated impact are calculated only when their source fields or explicit assumptions are available.

### 6.3 Statistical Testing

For binary outcomes, the platform uses a two-proportion z-test when pooled expected cell counts meet its threshold; for sparse tables it uses Fisher's exact test. Supported alternatives include two-sided and directional comparisons. These procedures assume valid independent observations in the compared groups. Repeated `user_id` or `session_id` values suppress independent-row inference; unique or absent identifiers do not prove independence.

### 6.4 Confidence Intervals and Effect Size

Wilson intervals are used for individual conversion rates, and the Newcombe-Wilson method is used for the difference in proportions. Effect-size outputs include the difference in proportions, relative risk, and odds ratio where estimable. These complement the p-value by showing the scale and uncertainty of an observed change.

### 6.5 Power and Minimum Detectable Effect

The planner uses a normal approximation for an independent two-group binary outcome with equal allocation and user-specified baseline rate, target rate, alpha, power, and sidedness. The dashboard's MDE uses the smaller valid group size as an equal-allocation approximation. These calculations support prospective planning; they do not establish that an existing experiment had adequate power. Estimates are unavailable for unsupported boundary rates or a zero target effect.

### 6.6 Segmentation, Anomaly Screening, and Guardrails

The application compares available segments and applies a Bonferroni adjustment within the analyzed dimension in the Detective workflow. Its configurable observation minimum is a low-volume screening cutoff, not proof of adequate power. Repeated identifiers suppress inferential alerts; repeated observations are not silently removed. Guardrails provide additional context so conversion improvements are not considered in isolation. Segment exploration and detector alerts are exploratory and do not independently prove causality.

### 6.7 Decision and Reporting

The rule-based decision engine combines the statistical result, uncertainty, effect size, sample information, available guardrails, and estimated business impact. The dashboard presents the evidence and supports report export. Uploaded-file metadata and a SHA-256 fingerprint can help identify the file used, but they do not verify its contents or business provenance.

## 7. Scope and Relevance

### In Scope

- Interactive analysis of a two-group A/B experiment with binary conversion outcomes.
- CSV and Excel ingestion, built-in/generated illustrative datasets, and validation.
- Conversion, lift, confidence interval, hypothesis test, effect-size, power/MDE, segmentation, and guardrail analysis.
- BusinessPulse Detective anomaly-screening workflow, charts, recommendations, and reports.
- Experiment and alert-status persistence through SQLite or configured PostgreSQL.

### Out of Scope and Limitations

- The application does not randomize users, collect production telemetry, or independently verify uploaded data.
- It does not prove causal validity, correct selection bias, or establish independence merely from unique identifiers.
- Paired, repeated-measures, clustered, and other non-independent designs require design-specific methods not implemented here.
- Multiple comparisons across separately explored dimensions are not jointly corrected by the within-dimension Bonferroni adjustment.
- The power planner is an approximation; detector sample thresholds are not power analyses.
- Authentication, role-based access, scheduled experimentation, and production data-source integrations are future extensions.

### Relevance

The project is relevant to product analytics, e-commerce, digital marketing, and software teams that need a repeatable way to inspect experiments. It demonstrates how data engineering, statistical reasoning, visualization, and business decision support can be integrated into an accessible application. Its explicit limitations also encourage responsible interpretation rather than overconfidence in automated results.

## 8. Objectives

1. Develop an interactive A/B testing and business analytics application using Python and Streamlit.
2. Provide an end-to-end workflow from data ingestion and validation to analysis, interpretation, and reporting.
3. Calculate conversion KPIs and compare Control and Treatment groups using appropriate statistical methods.
4. Communicate uncertainty and practical effect size through confidence intervals and effect metrics.
5. Support prospective experiment planning with power and minimum-detectable-effect estimates.
6. Incorporate segmentation, anomaly screening, guardrails, and business-impact context.
7. Persist experiment metadata and alert states while maintaining modular, testable components.
8. Make assumptions, data provenance limitations, and unsupported inference explicit to users.

## 9. Software Requirements

### 9.1 Functional Requirements

- Allow users to select sample, generated, synthetic demo, or uploaded CSV/Excel data.
- Validate required columns and report data-quality issues before analysis.
- Calculate conversion, lift, and available optional business metrics.
- Run supported tests, confidence intervals, and effect-size calculations.
- Provide power/MDE planning, segment analysis, anomaly screening, and guardrail checks.
- Present evidence-based recommendations and allow summary/report export.
- Store experiment, analysis, run, and alert-status records in the configured database.

### 9.2 Non-Functional Requirements

- **Usability:** provide an interactive dashboard for technical and business users.
- **Modularity:** separate ingestion, validation, analytics, visualization, persistence, and reporting.
- **Reliability:** validate inputs and label non-estimable results rather than presenting invalid inference as success.
- **Transparency:** surface assumptions, uncertainty, provenance limitations, and demo-data warnings.
- **Portability:** run locally with SQLite and support PostgreSQL through configuration.
- **Maintainability:** use automated tests for key analytical and database behavior.

### 9.3 Software and Runtime

- Operating system: Windows, macOS, or Linux with a supported Python installation.
- Python: 3.11 or later, per project documentation.
- Dependencies: Streamlit, Pandas, NumPy, SciPy, Statsmodels, Plotly, openpyxl, psycopg[binary] (optional PostgreSQL use), and Pytest. Version constraints are maintained in `requirements.txt`.
- Database: SQLite by default; PostgreSQL requires a configured `DATABASE_URL`.
- Input formats: CSV, XLSX, and XLS.
- Testing: Pytest.

### 9.4 Hardware (Indicative)

For local demonstration with the bundled sample data, a dual-core processor, 4 GB RAM, and sufficient free disk space for Python dependencies and uploaded files are reasonable baseline recommendations. Larger files and generated datasets may require additional memory and processing capacity. These are practical recommendations, not measured minimums or benchmark guarantees.

## 10. Conclusion

The Interactive A/B Testing and Business Analytics Platform combines data validation, statistical analysis, business metrics, visualizations, and decision support in one modular application. It is intended to make experiment results easier to inspect and explain while preserving important caveats about independence, power, data provenance, and causality. The system supports better-informed decisions, but final business decisions still require sound experiment design and domain judgment.

## 11. References

1. Kohavi, R., Deng, A., Frasca, B., Walker, T., Xu, Y., and Pohlmann, N. “Online Controlled Experiments at Large Scale.” *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '13)*, 2013, pp. 1168–1176. https://doi.org/10.1145/2487575.2488217
2. Kohavi, R., Tang, D., and Xu, Y. *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press, 2020.
3. Newcombe, R. G. “Interval Estimation for the Difference between Independent Proportions: Comparison of Eleven Methods.” *Statistics in Medicine*, vol. 17, no. 8, 1998, pp. 873–890. https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C873::AID-SIM779%3E3.0.CO;2-I
4. SciPy Developers. “scipy.stats.fisher_exact.” *SciPy Reference Guide*. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
5. Statsmodels Developers. “statsmodels.stats.proportion.proportions_ztest.” *Statsmodels Documentation*. https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html
6. Streamlit, Inc. *Streamlit Documentation*. https://docs.streamlit.io/
