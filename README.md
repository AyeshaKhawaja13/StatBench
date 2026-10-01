# Statistical Reasoning Benchmark for LLMs

> **A reproducible benchmark for evaluating Large Language Models on statistical reasoning, inference, interpretation, regression diagnostics, and Bayesian reasoning.**

## Overview

This project evaluates how reliably a Large Language Model solves structured statistics problems across five core areas:

* Hypothesis Testing
* Confidence Intervals
* p-value Interpretation
* Regression Assumptions
* Bayes' Theorem

The benchmark contains **20 carefully designed problems**, with **4 problems per topic** and a mixture of easy, medium, and difficult questions.

The goal is not simply to measure whether an LLM produces the correct final answer, but to identify **where and why statistical reasoning fails**.

> **Accuracy and methodological honesty are prioritized over presentation or model performance claims.**

---

## Evaluation Result

| Metric                         |            Result |
| ------------------------------ | ----------------: |
| Problems Evaluated             |                20 |
| Correct Answers                |            **19** |
| Incorrect Answers              |             **1** |
| Overall Accuracy               |         **95.0%** |
| 95% Wilson Confidence Interval | **76.4% – 99.1%** |

> **Important:** The benchmark contains only 20 problems, so the confidence interval is necessarily wide. The result should not be interpreted as a stable estimate of general statistical reasoning ability.

---

## Benchmark Structure

### A. Hypothesis Testing — 4 Problems

Covers:

* One-sample and two-sample z-tests
* One-sample and two-sample t-tests
* Paired t-tests
* Chi-square tests
* Proportion tests
* One- vs two-tailed decisions
* Known vs unknown variance

### B. Confidence Intervals — 4 Problems

Covers:

* Mean confidence intervals
* Proportion confidence intervals
* Difference between means
* t vs z selection
* Small-sample inference
* Unequal variance situations

### C. p-value Interpretation — 4 Problems

Conceptual questions targeting common statistical misconceptions, including:

* What a p-value actually represents
* What a p-value does not represent
* Relationship between p-values and hypotheses
* Correct statistical conclusions

### D. Regression Assumptions — 4 Problems

Problems evaluate interpretation of diagnostics involving:

* Linearity
* Independence
* Homoscedasticity
* Normality of residuals
* Multicollinearity
* Residual plots and diagnostic patterns

### E. Bayes' Theorem — 4 Problems

Covers:

* Diagnostic testing
* Base-rate effects
* Conditional probability
* Posterior probabilities
* Multi-hypothesis updates
* Base-rate neglect

---

## Difficulty Design

The benchmark intentionally includes approximately:

* **⅓ Easy**
* **⅓ Medium**
* **⅓ Hard**

Difficult questions include common reasoning traps such as:

* Using z instead of t
* Treating paired observations as independent
* Ignoring unequal variances
* Misreading statistical assumptions
* Confusing p-values with the probability that the null hypothesis is true
* Base-rate neglect in diagnostic testing

Each problem also contains an internal **trap tag** identifying the expected failure mode.

---

## Evaluation Methodology

The benchmark follows a four-phase evaluation pipeline.

### Phase 1 — Problem Generation

Exactly 20 self-contained statistics problems are created.

Every problem specifies:

* Required statistical method
* Significance level (`α`)
* One- or two-tailed procedure
* Known or unknown variance where relevant
* Required rounding
* Complete input data
* Expected answer format
* Difficulty
* Potential reasoning trap

Each question is designed to have **one defensible answer**.

### Phase 1b — Reference Solutions

Every problem receives a detailed reference solution containing:

1. Given information
2. Method selection
3. Assumption checks
4. Numerical computation
5. Statistical conclusion
6. Independent self-audit

Numerical results are verified programmatically using scientific Python libraries such as:

```text
NumPy
SciPy
Statsmodels
```

The benchmark does not rely on mental arithmetic for reference answers.

---

## Phase 2 — Blind LLM Evaluation

The model is evaluated without access to:

* Reference solutions
* Expected answers
* Trap tags
* Grading information

Each problem is evaluated independently using the prompt:

```text
Solve the following statistics problem.
Show your reasoning, then give a final answer on a line
starting 'FINAL ANSWER:'.
```

The raw model response is preserved for later grading.

The evaluation records model metadata such as:

* Model name
* Model version
* Temperature
* Tool/code availability

---

## Phase 3 — Automated & Structured Grading

Only the model's **FINAL ANSWER** is compared against the reference key.

Numeric answers are evaluated according to the rounding specified by each problem.

For incorrect answers, exactly one primary error category is assigned.

### Error Categories

| Error Type             | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| Calculation            | Correct method/setup but incorrect arithmetic or rounding |
| Wrong test chosen      | Incorrect statistical procedure                           |
| Misread assumption     | Incorrect interpretation of a stated condition            |
| Misinterpreted p-value | Incorrect meaning or conclusion involving a p-value       |
| Other                  | Used only when no previous category applies               |

The grading follows an **earliest-error rule**: when multiple issues exist, the earliest error that caused the incorrect result is selected.

---

## Phase 4 — Reporting

The evaluation generates a statistical report containing:

* Overall accuracy
* 95% Wilson confidence interval
* Error rate by topic
* Error counts by error type
* Important observed failure patterns
* Problem IDs supporting each finding
* Hypotheses explaining potential failure mechanisms
* Study limitations

---

## Output Files

The benchmark produces the following artifacts:

```text
├── problems.json
├── solutions.md
├── llm_responses.json
├── grading.csv
└── report.md
```

### `problems.json`

Contains the complete benchmark questions, metadata, difficulty levels, and trap tags.

### `solutions.md`

Contains the verified reference solutions and computational checks.

### `llm_responses.json`

Stores the raw, unedited LLM responses.

### `grading.csv`

Contains structured evaluation results:

```text
id
topic
difficulty
key
llm_answer
correct
error_type
justification
```

### `report.md`

Contains the final benchmark analysis and statistical summary.

---

## Current Result

The current evaluation produced:

```text
Evaluation Complete

Overall Accuracy: 19/20 (95.0%)

95% Wilson Confidence Interval:
[76.4%, 99.1%]

Total Problems: 20
Correct: 19
Incorrect: 1
```

Because **n = 20** is small, the Wilson interval demonstrates substantial uncertainty around the observed accuracy.

The result should therefore be treated as an **initial benchmark measurement**, rather than a definitive estimate of the model's statistical reasoning capability.

---

## Quality Control

Before considering an evaluation complete, the following checks are performed:

* [x] 20 problems created
* [x] 4 problems per topic
* [x] Numerical reference answers verified computationally
* [x] Problems designed to have one defensible answer
* [x] Blind evaluation methodology defined
* [x] Raw LLM responses preserved
* [x] Incorrect answers assigned one primary error type
* [x] Grading output generated
* [x] Wilson confidence interval reported
* [x] Report generated from grading results

---

## Reproducibility

To reproduce the benchmark:

1. Clone the repository.
2. Install the required Python dependencies.
3. Configure your own LLM/API credentials locally.
4. Run the benchmark generation and evaluation pipeline.
5. Review the generated artifacts.
6. Inspect `grading.csv` and `report.md`.

### API Key Setup

**Never commit your API key to GitHub.**

Create a local `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Add it to `.gitignore`:

```gitignore
.env
.env.local
.env.*.local
```

Anyone reproducing the benchmark should use **their own API key**.

---

## Why This Benchmark?

LLMs can produce convincing statistical explanations while still making fundamental mistakes in:

* Test selection
* Statistical assumptions
* Probability interpretation
* Numerical computation
* Bayesian reasoning

A high-level accuracy score alone does not reveal these failure modes.

This benchmark therefore combines **answer accuracy + error classification + statistical analysis** to provide a more informative evaluation.

---

## Limitations

The current evaluation has several important limitations:

1. **Small sample size:** only 20 problems were evaluated.
2. **Single evaluation run:** repeated runs may produce different answers.
3. **Problem-specific effects:** benchmark wording can influence model performance.
4. **Limited domain coverage:** the benchmark represents selected statistics concepts rather than statistics as a whole.
5. **Potential benchmark ambiguity:** even carefully reviewed questions may contain edge cases or interpretation effects.

For stronger conclusions, the benchmark should be rerun with **at least 3 independent samples per problem** and preferably across multiple models.

---

## Future Work

Planned extensions include:

* Increasing the benchmark from 20 to 100+ problems
* Running multiple samples per problem
* Comparing multiple LLMs
* Measuring reasoning consistency
* Evaluating tool-assisted vs non-tool-assisted performance
* Adding effect-size and power analysis questions
* Adding ANOVA and non-parametric tests
* Adding time-series diagnostics
* Comparing zero-shot and few-shot prompting
* Statistical significance testing between model performances

---

## Tech Stack

```text
Python
NumPy
SciPy
Statsmodels
JSON
CSV
Markdown
LLM APIs
```

---

## Project Philosophy

> **Do not optimize the benchmark to make the model look good. Optimize it to reveal what the model gets wrong.**

The benchmark is designed around reproducibility, transparent grading, computational verification, and honest reporting of uncertainty.

---

## Author

**Ayesha Shafique**

AI Engineer | Machine Learning | Agentic AI

This project explores the intersection of **AI evaluation, statistical reasoning, and reliable LLM systems**.
