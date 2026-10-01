# Statistics Benchmark Evaluation Report

## 1. Executive Summary & Overall Accuracy
A 20-problem benchmark evaluating core undergraduate and graduate statistical competency was administered blindly to **OpenAI GPT-OSS-20B** (via Groq Cloud API, temperature $T = 0.0$, tools/code execution disallowed). The model achieved:
- **Overall Accuracy**: **18 / 20 (90.0%)**
- **95% Wilson Score Confidence Interval**: **[69.9%, 97.2%]**

> **Note on Sample Size**: Because $n = 20$ is small, the Wilson interval spans nearly 27 percentage points. A high point estimate of 90% must be interpreted cautiously, as small samples cannot rule out moderate underlying failure rates.

---

## 2. Topic Error Rates & Error Type Breakdown

### Table 1: Error Rate by Topic (A–E)
| Topic | Topic Description | Tested Problems | Correct | Errors | Topic Error Rate |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **A** | Hypothesis testing | P01, P02, P03, P04 | 3 | 1 | **25.0%** |
| **B** | Confidence intervals | P05, P06, P07, P08 | 3 | 1 | **25.0%** |
| **C** | p-value interpretation | P09, P10, P11, P12 | 4 | 0 | **0.0%** |
| **D** | Regression assumptions | P13, P14, P15, P16 | 4 | 0 | **0.0%** |
| **E** | Bayes' theorem | P17, P18, P19, P20 | 4 | 0 | **0.0%** |
| **Total** | **All 5 Topics** | **20 Problems** | **18** | **2** | **10.0%** |

### Table 2: Primary Error Type Counts
| Primary Error Type | Count | Affected Problem IDs | Description |
| :--- | :---: | :--- | :--- |
| **Calculation** | **2** | P01, P08 | Right method and theoretical setup, but intermediate arithmetic error or premature rounding. |
| **Wrong test chosen** | **0** | None | Appropriately distinguished paired vs independent, Welch vs pooled, and z vs t. |
| **Misread assumption** | **0** | None | Adhered to stated $\alpha$, unpooled bounds, and diagnostic polarity. |
| **Misinterpreted p-value** | **0** | None | Perfectly identified frequentist definitions and avoided fallacies. |
| **Other** | **0** | None | No base rate neglect or uncategorized conceptual failures observed. |

---

## 3. Three Most Important Findings

1. **Vulnerability to Multi-Step Arithmetic Drift (`P01`)**:
   - In P01 (one-sample proportion z-test), the model correctly identified the null standard error formula $SE_0 = \sqrt{p_0(1-p_0)/n} = \sqrt{0.00011875}$. However, it made a mental arithmetic calculation error, computing $\sqrt{0.00011875} \approx 0.010889$ instead of $0.010897$, propagating into $z = 1.836$ instead of **1.835**.
   - *Hypothesis*: Without a code interpreter, sub-token floating-point arithmetic is approximated via token probability distributions, leading to precision loss in non-integer square roots.
2. **Premature Intermediate Rounding Truncation (`P08`)**:
   - In P08 (difference in proportions CI), the model correctly set up the unpooled variance $SE = \sqrt{0.0019296} = 0.0439272$. However, it rounded $SE$ prematurely to $0.0439$, resulting in $ME = 1.645 \times 0.0439 = 0.0722155 \to 0.0722$ rather than **0.0723**.
   - *Hypothesis*: The model treats intermediate quantities as final decimal representations rather than carrying full precision through the calculation chain.
3. **Flawless Conceptual Reasoning Across P-Values and Bayes (`P09`–`P20`)**:
   - The model demonstrated 100% accuracy on conceptual p-value items (distinguishing statistical vs practical significance and avoiding the inverse probability fallacy) and successfully resisted base-rate neglect in rare disease screening (`P17`, `P18`), sequential Bayes updates, and MAP selection (`P20`).

---

## 4. Limitations & Recommendations
- **Sample Size ($n = 20$)**: High variance in accuracy bounds.
- **Single-Sample Evaluation**: Temperature $0.0$ on a single run. We recommend repeating evaluation with $K \ge 3$ runs across multiple open/closed models using `run_eval.py` to test stability.
