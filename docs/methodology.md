# Statistical Methodology

## Conversion Metric

The primary metric used in the application is conversion rate:

Conversion Rate = Conversions / Total Users

## Lift Measures

- Absolute Lift = Treatment CR - Control CR
- Relative Lift = (Treatment CR - Control CR) / Control CR * 100

## Hypothesis Testing

For binary conversion outcomes, the application uses a two-proportion z-test when pooled expected cell counts are at least five and Fisher's exact test for sparse tables. The alternatives are two-sided or directional (Treatment greater/less than Control). Tests use complete valid 0/1 outcomes and assume independent observations from two groups.

## Confidence Intervals

The application calculates Wilson confidence intervals for:

- Control conversion rate
- Treatment conversion rate
- Treatment-minus-Control difference using the Newcombe-Wilson method

Confidence intervals are shown with the selected alpha level (90%, 95%, or 99%).

## Effect Size

Effect size complements p-value:

- Difference in proportions
- Relative risk
- Odds ratio

This helps interpret whether the observed effect is meaningful in business terms, not just statistically detectable.

## Statistical Power and MDE

The planner uses a normal approximation for a two-group binary conversion outcome with independent randomized units, equal allocation, specified baseline and target treatment rates, alpha, desired power, and sidedness. Required sample size and MDE are unavailable for zero target effects and for boundary rates unsupported by this approximation. The dashboard MDE uses the smaller valid group size as an equal-allocation approximation.

These are planning estimates, not post-hoc evidence that an existing experiment achieved adequate power. They do not verify randomization, independence, or the model assumptions for uploaded data. The configured Detective minimum observation count is a low-volume screening rule, not a power analysis.

Repeated `user_id` or `session_id` values suppress the independent-row A/B test and Detective inferential alerts. Exact duplicate rows are retained in descriptive calculations; possible repeated observations suppress Detective inferential alerts rather than being automatically removed. Unique identifiers, or the absence of repeated identifiers, do not establish independence; paired, repeated-measures, or clustered designs require a design-specific method not implemented here. When the design is unknown, results remain conditional on the independent-row assumption.

Uploaded-file metadata includes its filename, upload time, row/column counts, inferred column types, missing-value counts, file size, and SHA-256 byte fingerprint for the current session and report export. The fingerprint only helps identify identical bytes; it does not verify file contents or business origin. Uploaded business provenance remains unverified.

## Guardrails

Guardrail metrics protect the business from recommending a treatment solely because conversion improves. Examples include:

- revenue per user
- bounce rate
- error rate
- page views per session
- session duration

## Decision Logic

The recommendation engine combines:

- p-value
- confidence interval sign
- effect size
- sample size
- guardrail checks
- estimated business impact

This avoids misleading conclusions based only on significance thresholds.

## Important Limitations

- Data quality affects all conclusions.
- Small samples can produce unstable estimates.
- Multiple comparisons increase false positive risk.
- p-value thresholds should not be treated as automatic business decisions.
- Repeated observations per randomized unit and clustered assignment violate the independent-row assumption unless modeled explicitly.
- Business fields are optional; a KPI is shown only when its source column is present. Unknown fields are not imputed as business results.
