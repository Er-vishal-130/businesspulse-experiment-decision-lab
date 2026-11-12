# Viva Questions and Answers

## 1. Why did you choose this project?

I chose this project because it combines multiple core areas of computer science and data analytics. It includes Python, statistics, SQL, data processing, visualization, web application design, and business decision-making in a single end-to-end system.

## 2. What is the main objective of your project?

The main objective is to build a platform that helps users analyze A/B experiments using rigorous statistical reasoning and business context, instead of relying on superficial dashboard metrics alone.

## 3. What is A/B testing?

A/B testing is a controlled experiment where two versions, usually control and treatment, are compared to determine whether a change leads to a statistically and practically meaningful improvement.

## 4. What is the difference between statistical significance and business significance?

Statistical significance addresses whether the observed difference is unlikely to be random. Business significance addresses whether the impact is large enough to matter for revenue, conversion, or operations.

## 5. What does p-value mean?

The p-value is the probability of observing data at least as extreme as the experiment if the null hypothesis were true. Lower p-values suggest stronger evidence against the null hypothesis.

## 6. What happens if p-value is greater than 0.05?

It means the result is not statistically significant at the selected alpha level, so we may fail to reject the null hypothesis. This does not prove there is no effect; it means the evidence is insufficient.

## 7. What is the null hypothesis in this project?

The null hypothesis states that there is no difference between the control and treatment groups.

## 8. What is the alternative hypothesis?

The alternative hypothesis states that there is a measurable difference between the control and treatment groups.

## 9. What is absolute lift?

Absolute lift is the difference between the treatment conversion rate and the control conversion rate.

## 10. What is relative lift?

Relative lift measures the percentage change in conversion relative to the control group.

## 11. What is statistical power?

Power is the probability of correctly detecting a real effect when it exists.

## 12. What is MDE?

MDE stands for minimum detectable effect. It is the smallest effect size the experiment is designed to detect reliably.

## 13. Why are guardrail metrics important?

Guardrail metrics help prevent a business from choosing a treatment that improves conversion but harms user experience, revenue quality, or product health.

## 14. What are guardrail metrics in this project?

Examples include revenue per user, bounce rate, error rate, average session duration, and pages per session.

## 15. What is the purpose of confidence intervals?

Confidence intervals quantify uncertainty around an estimate and show a plausible range for the true effect.

## 16. What is effect size?

Effect size measures the magnitude of an observed difference, not just whether it is statistically significant.

## 17. Why does the project avoid using p-value alone for decision-making?

Because statistical significance does not guarantee business value. Sample size, effect size, uncertainty, and guardrails must also be considered.

## 18. Which libraries and databases are used in the project?

Python, Streamlit, Pandas, NumPy, SciPy, Statsmodels, Plotly, SQLite, and PostgreSQL are used.

## 19. Why did you use Streamlit?

Streamlit provides a fast way to build interactive web apps with Python without building a full front-end framework.

## 20. How does the project support SQLite and PostgreSQL?

SQLite is the default for simple local use. For PostgreSQL, the application reads a `DATABASE_URL` environment variable and uses the psycopg driver; database tables are initialized automatically.

## 21. What is the role of the database in the project?

The database stores experiment metadata and analytical results so the project has a structured persistence layer.

## 22. What makes this project better than a normal dashboard?

It goes beyond charts by including data validation, statistical testing, guardrail analysis, business interpretation, and recommendation logic.

## 23. What happens if the p-value is less than alpha?

The result is statistically significant at the chosen threshold, and the null hypothesis may be rejected.

## 24. Can p-value less than 0.05 always mean the treatment is good?

No. It may still be operationally risky or too small to matter in business terms.

## 25. What is the significance of sample size in A/B testing?

A large sample reduces variability and makes the experiment more reliable. Small sample sizes can create unstable estimates.

## 26. Why is multiple testing a risk?

Running many comparisons increases the chance of false discoveries. This is why segment-level outputs should be interpreted cautiously.

## 27. What is the peeking problem?

The peeking problem occurs when an experiment is repeatedly checked and stopped early after crossing significance thresholds, which can inflate false positives.

## 28. What is the workflow of your application?

The project follows: Data → Validation → KPI → Statistical Analysis → Uncertainty → Segmentation → Guardrails → Business Impact → Decision.

## 29. What is your contribution in this project?

My contribution includes designing the architecture, implementing the analysis pipeline, integrating the statistical methods, building the dashboard, and creating documentation for the project.

## 30. What did you learn from this project?

I gained practical knowledge of statistical analysis, business decision-making, Python application design, and how analytical systems should balance technical correctness with stakeholder usability.
