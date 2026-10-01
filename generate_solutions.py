import json
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

with open("problems.json", "r", encoding="utf-8") as f:
    problems = json.load(f)

# Verification & Dual Derivations
solutions_md = """# Statistics Benchmark Reference Solutions & Dual-Method Self-Audit

This document contains the official reference solutions for all 20 problems in the statistics benchmark.
Every problem follows a strict 5-part structure:
1. **Given Information**
2. **Method Choice & Justification**
3. **Assumption Checks**
4. **Step-by-Step Computation & Python Verification**
5. **Conclusion in Context**
6. **Self-Audit / Dual-Method Re-Derivation**

---
"""

# Let's craft the solution details for each problem
problem_solutions = [
    # P01
    {
        "id": "P01",
        "title": "One-sample z-test for a population proportion",
        "given": """- Null hypothesis: $H_0: p \le 0.05$ (or $p = 0.05$)
- Alternative hypothesis: $H_1: p > 0.05$ (one-tailed, upper-tail test)
- Sample size: $n = 400$
- Number of defectives: $x = 28$
- Sample proportion: $\hat{p} = \frac{28}{400} = 0.07$
- Significance level: $\alpha = 0.05$
- Standard normal approximation without continuity correction""",
        "method": "One-sample z-test for proportions using the null proportion $p_0$ to compute standard error (Score test). In hypothesis testing for a single proportion, standard error is evaluated under the null hypothesis: $SE_0 = \sqrt{\frac{p_0(1-p_0)}{n}}$.",
        "assumptions": """1. **Random sampling**: The 400 microchips constitute a simple random sample.
2. **Independence**: $n = 400$ is much less than 10% of total plant production ($10\% \text{ condition}$).
3. **Success/Failure condition**: $n p_0 = 400 \times 0.05 = 20 \ge 10$ and $n(1 - p_0) = 400 \times 0.95 = 380 \ge 10$. Both conditions are satisfied, justifying normal approximation.""",
        "computation": """```python
import numpy as np
import scipy.stats as stats

p0 = 0.05
n = 400
x = 28
p_hat = x / n # 0.07

se_0 = np.sqrt(p0 * (1 - p0) / n) # sqrt(0.05 * 0.95 / 400) = 0.010897247...
z = (p_hat - p0) / se_0 # (0.07 - 0.05) / 0.010897247 = 1.8353258...
p_value = 1 - stats.norm.cdf(z) # 0.0332
```
Calculation details:
$$SE_0 = \sqrt{\frac{0.05 \times 0.95}{400}} = \sqrt{\frac{0.0475}{400}} = \sqrt{0.00011875} \approx 0.01089725$$
$$z = \frac{0.07 - 0.05}{0.01089725} = \frac{0.02}{0.01089725} \approx 1.83533$$
Rounding to 3 decimal places gives **1.835**.""",
        "conclusion": "The test statistic is $z = 1.835$. Because $z = 1.835 > z_{0.05} = 1.645$ (corresponding to $p = 0.0332 < 0.05$), we reject the null hypothesis at $\alpha = 0.05$ and conclude there is sufficient evidence that the defect rate exceeds 5%.",
        "audit": """**Method 2 (Statsmodels `proportions_ztest` / Exact Binomial Check)**:
```python
from statsmodels.stats.proportion import proportions_ztest, binom_test
# Score z-test:
z_sm, p_sm = proportions_ztest(count=28, nobs=400, value=0.05, alternative='larger', prop_var=0.05)
print(f"Statsmodels z: {z_sm:.3f}, p-value: {p_sm:.4f}")
# Exact Binomial:
p_exact = stats.binomtest(28, 400, 0.05, alternative='greater').pvalue
print(f"Exact Binomial p-value: {p_exact:.4f}")
```
Output: `z = 1.835`, `p = 0.0332`. Both analytic and statsmodels evaluations match exactly at **1.835**."""
    },
    # P02
    {
        "id": "P02",
        "title": "Paired t-test for dependent samples",
        "given": """- Sample size: $n = 10$ patients
- Paired differences ($d_i = \text{After}_i - \text{Before}_i$): $[-6, -4, -8, -5, -7, -3, -9, -4, -6, -8]$
- Null hypothesis: $H_0: \mu_d = 0$
- Alternative hypothesis: $H_1: \mu_d \neq 0$ (two-tailed)
- Significance level: $\alpha = 0.01$
- Population differences are approximately normally distributed.""",
        "method": "Paired-samples Student's t-test. The measurements are repeated before-and-after observations on the same 10 individuals, creating within-subject pairing. The test reduces to a one-sample t-test on the differences $d_i$: $t = \frac{\bar{d} - 0}{s_d / \sqrt{n}}$ with $df = n - 1 = 9$.",
        "assumptions": """1. **Paired observations**: Each pair of measurements corresponds to the same subject.
2. **Independence across subjects**: Patient responses are independent across the sample.
3. **Normality of differences**: The distribution of differences $d_i$ is stated to be approximately normal.""",
        "computation": """```python
import numpy as np
import scipy.stats as stats

d = np.array([-6, -4, -8, -5, -7, -3, -9, -4, -6, -8], dtype=float)
n = len(d) # 10
mean_d = np.mean(d) # -6.0
s_d = np.std(d, ddof=1) # 2.0
se_d = s_d / np.sqrt(n) # 2.0 / sqrt(10) = 0.6324555...
t_stat = mean_d / se_d # -6.0 / 0.6324555 = -9.48683...
p_val = 2 * stats.t.cdf(t_stat, df=n-1) # 5.64e-6
```
Calculation details:
$$\sum d_i = -60 \implies \bar{d} = -6.0$$
$$\sum (d_i - \bar{d})^2 = 0 + 4 + 4 + 1 + 1 + 9 + 9 + 4 + 0 + 4 = 36 \implies s_d = \sqrt{\frac{36}{9}} = \sqrt{4} = 2.0$$
$$SE = \frac{2.0}{\sqrt{10}} \approx 0.6324555$$
$$t = \frac{-6.0}{0.6324555} = -3\sqrt{10} \approx -9.48683$$
Rounding to 3 decimal places gives **-9.487**.""",
        "conclusion": "The test statistic is $t = -9.487$ ($df = 9, p = 5.64 \times 10^{-6}$). Since $|t| = 9.487 > t_{0.005, 9} = 3.250$, we reject $H_0$ at $\alpha = 0.01$ and conclude the drug causes a statistically significant reduction in systolic blood pressure.",
        "audit": """**Method 2 (`scipy.stats.ttest_1samp` / manual sum of squares)**:
```python
t_check, p_check = stats.ttest_1samp(d, 0)
print(f"Direct t-test: t = {t_check:.4f}, exact value = {-3*np.sqrt(10):.4f}")
```
Both yield $t = -3\sqrt{10} \approx -9.48683$, rounding to **-9.487**."""
    },
    # P03
    {
        "id": "P03",
        "title": "Welch's two-sample t-test with unequal variances",
        "given": """- Sample 1: $n_1 = 12, \bar{x}_1 = 24.5, s_1 = 1.8$
- Sample 2: $n_2 = 25, \bar{x}_2 = 21.0, s_2 = 4.2$
- Population variances: $\sigma_1^2 \neq \sigma_2^2$ (unequal / unpooled)
- Hypotheses: $H_0: \mu_1 - \mu_2 = 0$ vs $H_1: \mu_1 - \mu_2 \neq 0$
- Rounding: 3 decimal places""",
        "method": "Welch's t-test (unequal variances t-test). When population variances cannot be assumed equal ($s_2^2 / s_1^2 = 4.2^2 / 1.8^2 = 17.64 / 3.24 \approx 5.44$), pooling is inappropriate. The standard error is: $SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$, and the test statistic is $t = \frac{\bar{x}_1 - \bar{x}_2}{SE}$.",
        "assumptions": """1. **Independent random samples**: Two distinct plots randomized to soil treatments.
2. **Normality**: Crop yields within plots are normally distributed.
3. **Heteroscedasticity**: $\sigma_1^2 \neq \sigma_2^2$, requiring Welch's formulation without pooling.""",
        "computation": """```python
import numpy as np
import scipy.stats as stats

n1, m1, s1 = 12, 24.5, 1.8
n2, m2, s2 = 25, 21.0, 4.2

v1 = (s1**2) / n1 # 3.24 / 12 = 0.27
v2 = (s2**2) / n2 # 17.64 / 25 = 0.7056
se_diff = np.sqrt(v1 + v2) # sqrt(0.9756) = 0.987724658...
t_stat = (m1 - m2) / se_diff # 3.5 / 0.987724658 = 3.543497...

# Welch-Satterthwaite df:
df_num = (v1 + v2)**2
df_den = (v1**2)/(n1 - 1) + (v2**2)/(n2 - 1)
df = df_num / df_den # 34.7728
```
Calculation details:
$$\frac{s_1^2}{n_1} = \frac{1.8^2}{12} = \frac{3.24}{12} = 0.2700$$
$$\frac{s_2^2}{n_2} = \frac{4.2^2}{25} = \frac{17.64}{25} = 0.7056$$
$$SE = \sqrt{0.2700 + 0.7056} = \sqrt{0.9756} \approx 0.9877247$$
$$t = \frac{24.5 - 21.0}{0.9877247} = \frac{3.5}{0.9877247} \approx 3.54350$$
Rounding to 3 decimal places gives **3.543**.""",
        "conclusion": "The Welch t-statistic is $t = 3.543$ ($df = 34.77, p = 0.0011$). At any standard significance level ($\alpha = 0.05$ or $0.01$), we reject the null hypothesis and conclude that the soil treatments produce significantly different mean crop yields.",
        "audit": """**Method 2 (`scipy.stats.ttest_ind_from_stats` with `equal_var=False`)**:
```python
t_check, p_check = stats.ttest_ind_from_stats(
    mean1=24.5, std1=1.8, nobs1=12,
    mean2=21.0, std2=4.2, nobs2=25,
    equal_var=False
)
print(f"Scipy Welch t: {t_check:.4f}, p: {p_check:.6f}")
```
Output: `t = 3.5435`, `p = 0.001149`. Matches **3.543** perfectly."""
    },
    # P04
    {
        "id": "P04",
        "title": "Pearson's Chi-square test of independence",
        "given": """- $2 \times 2$ Contingency Table:
  - Row 1 (Treatment A): 40 Recovered, 60 Not Recovered (Total = 100)
  - Row 2 (Treatment B): 60 Recovered, 40 Not Recovered (Total = 100)
- Column totals: Recovered = 100, Not Recovered = 100
- Grand Total: $N = 200$
- No Yates' continuity correction.""",
        "method": "Pearson's Chi-Square Test of Independence without continuity correction. For an $r \times c$ table, expected cell frequency is $E_{ij} = \frac{R_i C_j}{N}$, and test statistic is $\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$ with $df = (r-1)(c-1) = 1$.",
        "assumptions": """1. **Categorical data**: Both treatment and recovery are binary categorical variables.
2. **Independent observations**: Each patient is counted in exactly one cell.
3. **Expected frequencies**: All expected cell counts $E_{ij} = 50 \ge 5$, easily meeting Cochran's criterion.""",
        "computation": """```python
import numpy as np
import scipy.stats as stats

obs = np.array([[40, 60], [60, 40]])
chi2, p, dof, ex = stats.chi2_contingency(obs, correction=False)
```
Calculation details:
$$E_{11} = \frac{100 \times 100}{200} = 50, \quad E_{12} = 50, \quad E_{21} = 50, \quad E_{22} = 50$$
$$\chi^2 = \frac{(40 - 50)^2}{50} + \frac{(60 - 50)^2}{50} + \frac{(60 - 50)^2}{50} + \frac{(40 - 50)^2}{50}$$
$$\chi^2 = \frac{100}{50} + \frac{100}{50} + \frac{100}{50} + \frac{100}{50} = 2 + 2 + 2 + 2 = 8.000$$
Rounding to 3 decimal places gives **8.000**.""",
        "conclusion": "The test statistic is $\chi^2 = 8.000$ ($df = 1, p = 0.0047$). Since $\chi^2 = 8.000 > \chi^2_{0.05, 1} = 3.841$, we reject independence and conclude recovery rates differ significantly between treatments.",
        "audit": """**Method 2 (Two-proportion z-test relationship $\chi^2 = z^2$)**:
For a $2 \times 2$ table, Pearson $\chi^2$ equals the squared pooled two-proportion z-statistic:
$$\hat{p}_1 = 0.40, \hat{p}_2 = 0.60, \hat{p}_{\text{pool}} = \frac{40+60}{200} = 0.50$$
$$SE_{\text{pool}} = \sqrt{0.5 \times 0.5 \times (1/100 + 1/100)} = \sqrt{0.25 \times 0.02} = \sqrt{0.005} \approx 0.07071068$$
$$z = \frac{0.40 - 0.60}{0.07071068} = \frac{-0.20}{0.07071068} = -\sqrt{8} \approx -2.828427$$
$$z^2 = (-\sqrt{8})^2 = 8.000$$
Both methods match identically at **8.000**."""
    },
    # P05
    {
        "id": "P05",
        "title": "Confidence interval for normal mean with small n and unknown sigma",
        "given": """- Sample size: $n = 9$
- Sample mean: $\bar{x} = 50.0 \text{ mg/L}$
- Sample standard deviation: $s = 6.0 \text{ mg/L}$
- Population: Normally distributed with unknown variance $\sigma^2$
- Confidence level: $1 - \alpha = 0.95 \implies \alpha = 0.05$, two-sided""",
        "method": "Student's t confidence interval. Because the population standard deviation $\sigma$ is unknown and estimated by sample standard deviation $s$ with a small sample size ($n = 9$), the pivot $\frac{\bar{x} - \mu}{s/\sqrt{n}}$ follows Student's t distribution with $df = n - 1 = 8$.",
        "assumptions": """1. **Normality**: The underlying population is normally distributed.
2. **Random sampling**: Batches are produced independently under identical conditions.
3. **Unknown variance**: $\sigma$ is unknown, requiring Student's t critical value rather than normal $z$.""",
        "computation": """```python
import numpy as np
import scipy.stats as stats

n = 9
xbar = 50.0
s = 6.0
df = n - 1 # 8
t_crit = stats.t.ppf(0.975, df=df) # 2.306004...
se = s / np.sqrt(n) # 6.0 / 3 = 2.0
margin_of_error = t_crit * se # 2.306004 * 2.0 = 4.612008...
upper_limit = xbar + margin_of_error # 54.612008...
```
Calculation details:
$$SE = \frac{6.0}{\sqrt{9}} = \frac{6.0}{3} = 2.0$$
$$t_{0.025, 8} = 2.3060041$$
$$E = 2.3060041 \times 2.0 = 4.612008$$
$$\text{Upper Limit} = 50.0 + 4.612008 \approx 54.612$$
Rounding to 3 decimal places gives **54.612**.""",
        "conclusion": "We are 95% confident that the true population mean active compound concentration is between $45.388 \text{ mg/L}$ and $54.612 \text{ mg/L}$. The requested upper limit is **54.612**.",
        "audit": """**Method 2 (`scipy.stats.t.interval`)**:
```python
ci_low, ci_high = stats.t.interval(0.95, df=8, loc=50.0, scale=6.0/np.sqrt(9))
print(f"Scipy interval: [{ci_low:.4f}, {ci_high:.4f}]")
```
Output: `ci_high = 54.6120`. Matches **54.612** exactly."""
    },
    # P06
    {
        "id": "P06",
        "title": "Confidence interval for difference in means with known population variances",
        "given": """- Sample 1: $n_1 = 15, \bar{x}_1 = 78.0, \sigma_1^2 = 16.0 \implies \sigma_1 = 4.0$
- Sample 2: $n_2 = 20, \bar{x}_2 = 72.0, \sigma_2^2 = 25.0 \implies \sigma_2 = 5.0$
- Population variances are KNOWN
- Confidence level: $99\% \implies \alpha = 0.01$, two-sided $\alpha/2 = 0.005$""",
        "method": "Standard normal (z-distribution) confidence interval for two independent means with known population variances. Because $\sigma_1^2$ and $\sigma_2^2$ are known parameters, the exact sampling distribution of $\frac{(\bar{x}_1 - \bar{x}_2) - (\mu_1 - \mu_2)}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}}$ is standard normal $\mathcal{N}(0, 1)$, irrespective of small sample size.",
        "assumptions": """1. **Known variances**: Population variances $\sigma_1^2 = 16$ and $\sigma_2^2 = 25$ are true parameters, not sample estimates.
2. **Normality**: Both fill volume populations are normally distributed.
3. **Independence**: Independent random samples between machines.""",
        "computation": """```python
import numpy as np
import scipy.stats as stats

sig1_sq, n1 = 16.0, 15
sig2_sq, n2 = 25.0, 20
z_crit = stats.norm.ppf(0.995) # 2.5758293...
se = np.sqrt(sig1_sq / n1 + sig2_sq / n2) # sqrt(16/15 + 25/20) = sqrt(1.066667 + 1.25) = sqrt(2.316667) = 1.5220599...
margin_of_error = z_crit * se # 2.5758293 * 1.5220599 = 3.920566...
```
Calculation details:
$$SE = \sqrt{\frac{16}{15} + \frac{25}{20}} = \sqrt{\frac{16}{15} + \frac{5}{4}} = \sqrt{\frac{64 + 75}{60}} = \sqrt{\frac{139}{60}} \approx 1.5220599$$
$$z_{0.005} \approx 2.5758293$$
$$E = 2.5758293 \times 1.5220599 \approx 3.92057$$
Rounding to 3 decimal places gives **3.921**.""",
        "conclusion": "The margin of error for a 99% confidence interval of $\mu_1 - \mu_2$ is **3.921** mL.",
        "audit": """**Method 2 (`scipy.stats.norm.interval`)**:
```python
se_val = np.sqrt(139.0 / 60.0)
ci = stats.norm.interval(0.99, loc=0, scale=se_val)
me_scipy = ci[1]
print(f"Scipy norm margin of error: {me_scipy:.4f}")
```
Output: `me_scipy = 3.9206`, rounding to **3.921**."""
    },
    # P07
    {
        "id": "P07",
        "title": "Wald confidence interval lower bound for a single proportion",
        "given": """- Sample size: $n = 100$
- Number of successes: $x = 35$
- Sample proportion: $\hat{p} = \frac{35}{100} = 0.35$
- Confidence level: $95\% \implies \alpha = 0.05, z_{0.025} \approx 1.959964$
- Standard Wald confidence interval formula specified.""",
        "method": "Standard Wald confidence interval for a single proportion: $\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$.",
        "assumptions": """1. **Random sampling**: Survey respondents represent a random sample of voters.
2. **Success/Failure condition**: $n \hat{p} = 35 \ge 10$ and $n(1 - \hat{p}) = 65 \ge 10$, satisfying the empirical rule of thumb for asymptotic normality.""",
        "computation": """```python
import numpy as np
import scipy.stats as stats

n = 100
x = 35
p_hat = x / n # 0.35
z_crit = stats.norm.ppf(0.975) # 1.95996398...
se = np.sqrt(p_hat * (1 - p_hat) / n) # sqrt(0.35 * 0.65 / 100) = sqrt(0.002275) = 0.04769696...
margin_of_error = z_crit * se # 1.959964 * 0.04769696 = 0.0934843...
lower_bound = p_hat - margin_of_error # 0.35 - 0.0934843 = 0.2565157...
```
Calculation details:
$$SE = \sqrt{\frac{0.35 \times 0.65}{100}} = \sqrt{0.002275} \approx 0.04769696$$
$$E = 1.959964 \times 0.04769696 \approx 0.0934843$$
$$\text{Lower Bound} = 0.35 - 0.0934843 \approx 0.256516$$
Rounding to 4 decimal places gives **0.2565**.""",
        "conclusion": "The lower bound of the standard 95% Wald confidence interval for voter support is **0.2565** (25.65%).",
        "audit": """**Method 2 (`statsmodels.stats.proportion.proportion_confint` with `method='normal'`)**:
```python
from statsmodels.stats.proportion import proportion_confint
ci_low, ci_high = proportion_confint(count=35, nobs=100, alpha=0.05, method='normal')
print(f"Statsmodels Wald CI: [{ci_low:.6f}, {ci_high:.6f}]")
```
Output: `ci_low = 0.256516`. Matches **0.2565** exactly."""
    },
    # P08
    {
        "id": "P08",
        "title": "Confidence interval margin of error for difference in proportions",
        "given": """- Design A: $n_1 = 200, x_1 = 80 \implies \hat{p}_1 = \frac{80}{200} = 0.40$
- Design B: $n_2 = 250, x_2 = 60 \implies \hat{p}_2 = \frac{60}{250} = 0.24$
- Confidence level: $90\% \implies \alpha = 0.10, \alpha/2 = 0.05, z_{0.05} \approx 1.6448536$
- Unpooled Wald method for independent samples specified.""",
        "method": "Unpooled Wald confidence interval for the difference of two independent proportions: $E = z_{\alpha/2} \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}$.",
        "assumptions": """1. **Independent samples**: Visitors to Design A and Design B are independently randomized.
2. **Success/Failure condition**: $n_1 \hat{p}_1 = 80 \ge 10$, $n_1(1-\hat{p}_1) = 120 \ge 10$, $n_2 \hat{p}_2 = 60 \ge 10$, $n_2(1-\hat{p}_2) = 190 \ge 10$.
3. **No pooling**: In confidence interval estimation, proportions are unpooled because the true difference is not hypothesized to be zero.""",
        "computation": """```python
import numpy as np
import scipy.stats as stats

p1 = 80 / 200 # 0.40
p2 = 60 / 250 # 0.24
z_crit = stats.norm.ppf(0.95) # 1.6448536...
v1 = p1 * (1 - p1) / 200 # 0.24 / 200 = 0.0012
v2 = p2 * (1 - p2) / 250 # 0.1824 / 250 = 0.0007296
se = np.sqrt(v1 + v2) # sqrt(0.0019296) = 0.04392721...
margin_of_error = z_crit * se # 1.6448536 * 0.04392721 = 0.0722538...
```
Calculation details:
$$\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} = \frac{0.40 \times 0.60}{200} = \frac{0.24}{200} = 0.0012000$$
$$\frac{\hat{p}_2(1-\hat{p}_2)}{n_2} = \frac{0.24 \times 0.76}{250} = \frac{0.1824}{250} = 0.0007296$$
$$SE = \sqrt{0.0012000 + 0.0007296} = \sqrt{0.0019296} \approx 0.0439272$$
$$E = 1.6448536 \times 0.0439272 \approx 0.0722538$$
Rounding to 4 decimal places gives **0.0723**.""",
        "conclusion": "The margin of error for a 90% confidence interval of the conversion rate difference $(p_1 - p_2)$ is **0.0723** (7.23 percentage points).",
        "audit": """**Method 2 (`statsmodels.stats.proportion.confint_proportions_2indep`)**:
```python
from statsmodels.stats.proportion import confint_proportions_2indep
ci_low, ci_high = confint_proportions_2indep(80, 200, 60, 250, method='wald', alpha=0.10)
diff = (80/200) - (60/250) # 0.16
me_sm = (ci_high - ci_low) / 2.0
print(f"Statsmodels unpooled ME: {me_sm:.6f}")
```
Output: `me_sm = 0.072254`. Matches **0.0723** exactly."""
    },
    # P09
    {
        "id": "P09",
        "title": "Formal definition of a p-value",
        "given": """- Scenario: Medical trial reports two-tailed $p = 0.03$.
- 4 multiple-choice statements defining the p-value.""",
        "method": "Frequentist statistical definition analysis. By definition, a p-value is the probability, calculated under the assumption that the null hypothesis $H_0$ is true, of observing a test statistic as extreme as or more extreme than the observed sample outcome.",
        "assumptions": "Frequentist null hypothesis significance testing framework.",
        "computation": """Distractor Analysis:
- Option A: "There is a 3% probability that the null hypothesis is true." -> False. This commits the classic inverse probability fallacy: $P(H_0 \mid D) \neq P(\text{data} \mid H_0)$. Hypotheses are fixed parameters in frequentist statistics, not random variables with probabilities.
- Option B: "There is a 97% probability that the new medication is superior." -> False. Bayesian posterior probability cannot be inferred from a p-value without a prior distribution.
- Option C: "Assuming that the null hypothesis is true, the probability of observing a test statistic at least as extreme as the one calculated from the sample data is 0.03." -> Exactly the textbook mathematical definition of a p-value: $P(T \ge t_{\text{obs}} \mid H_0)$.
- Option D: "The probability of obtaining a false positive result if this experiment is replicated is 0.03." -> False. Replicability involves power and true effect size, not p-value.""",
        "conclusion": "Option C is the only mathematically and methodologically correct definition.",
        "audit": "ASA Statement on Statistical Significance and P-Values (Wasserstein & Lazar, 2016): 'Informally, a p-value is the probability under a specified statistical model that a statistical summary of the data would be equal to or more extreme than its observed value.' Option C is verified as the unique correct answer."
    },
    # P10
    {
        "id": "P10",
        "title": "Statistical significance versus practical significance",
        "given": """- Sample size: $n = 500,000$ office workers
- Measured effect: reduction in sedentary time of 42 seconds (95% CI: [38, 46] seconds)
- P-value: $p < 0.0001$
- Executive claim: "overwhelming practical health benefits and transforms workplace activity levels".""",
        "method": "Statistical vs Practical Significance Analysis. The standard error of an estimate scales as $\sigma / \sqrt{n}$. As $n \to \infty$, $SE \to 0$, causing any nonzero effect, however microscopic or practically meaningless, to yield an infinitesimal p-value.",
        "assumptions": "Large sample properties of test statistics.",
        "computation": """An average change of 42 seconds in a workday of 28,800 seconds (8 hours) represents a relative change of $42 / 28800 \approx 0.15\%$. While the narrow CI [38, 46] establishes that the 42-second reduction is precisely estimated and not a sampling artifact ($p < 0.0001$), 42 seconds is practically inconsequential for health outcomes.
- Option A asserts small p-values guarantee large effect magnitude (false).
- Option B correctly identifies that in huge samples, trivial effect sizes yield tiny p-values (true).
- Option C confuses estimation precision with clinical magnitude (false).
- Option D makes a bogus assertion about Gauss-Markov sample size limits (false).""",
        "conclusion": "Option B is the correct evaluation.",
        "audit": "Well-established in statistical literature (e.g., Lin et al., 2013 'Too Big to Fail: Large Samples and the p-Value Problem'). With $n = 500,000$, statistical power to detect trivial effect sizes approaches 1.0, rendering p-values uninformative about practical importance. Option B verified."
    },
    # P11
    {
        "id": "P11",
        "title": "Absence of evidence versus evidence of absence",
        "given": """- Sample size: $n = 12$ patients
- Test: two-tailed paired t-test
- Result: $t = 0.91, p = 0.38$
- Investigator conclusion: "Because $p > 0.05$, we accept the null hypothesis and conclude the compound produces zero arrhythmic side effects." """,
        "method": "Logic of Hypothesis Testing & Statistical Power. In frequentist hypothesis testing, failing to reject $H_0$ means data do not provide sufficient evidence against $H_0$; it never proves $H_0$ is true ('Absence of evidence is not evidence of absence').",
        "assumptions": "Statistical power with small sample size ($n=12$).",
        "computation": """With $n = 12$, a two-sample or paired t-test has very low statistical power to detect small or moderate effect sizes. A high p-value ($p = 0.38$) could easily occur even if the compound has real, dangerous side effects.
- Option A correctly states that failing to reject does not prove $H_0$ and points to low power from small sample size.
- Option B commits the inverse probability error ($P(H_1) = 0.38$).
- Option C is fabricated nonsense.
- Option D confuses one-tailed conversions.""",
        "conclusion": "Option A is the unique methodologically correct explanation.",
        "audit": "Standard statistical dogma (Altman & Bland 1995: 'Absence of evidence is not evidence of absence'). Equivalence testing (TOST) would be required to claim equivalence or zero effect, not failure to reject in a small underpowered sample. Option A verified."
    },
    # P12
    {
        "id": "P12",
        "title": "Multiple testing and family-wise error rate",
        "given": """- Number of independent tests: $m = 20$
- Nominal per-comparison error rate: $\alpha = 0.05$
- True state: all 20 null hypotheses are true ($H_{0,1}, \dots, H_{0,20}$ are true)
- No multiple testing adjustment applied.""",
        "method": "Family-Wise Error Rate (FWER) probability calculation under independence. For $m$ independent true null hypotheses, the probability of at least one false positive is: $\text{FWER} = 1 - (1 - \alpha)^m$.",
        "assumptions": "Independence across all 20 tests; all null hypotheses true.",
        "computation": """```python
alpha = 0.05
m = 20
p_no_type1 = (1 - alpha)**m # 0.95^20 = 0.3584859...
fwer = 1 - p_no_type1 # 1 - 0.3584859 = 0.641514...
```
Calculation details:
$$P(\text{No Type I errors}) = (1 - 0.05)^{20} = (0.95)^{20} \approx 0.358486$$
$$\text{FWER} = 1 - P(\text{No Type I errors}) = 1 - 0.358486 \approx 0.641514$$
Rounding to 3 decimal places gives **0.642**, which corresponds to Option C.""",
        "conclusion": "The probability of at least one false positive is 0.642 (Option C).",
        "audit": """**Method 2 (`scipy.stats.binom.sf`)**:
Using the Binomial survival function $P(X \ge 1)$ where $X \sim \text{Binomial}(20, 0.05)$:
```python
p_binom = stats.binom.sf(0, 20, 0.05)
print(f"Binomial SF: {p_binom:.6f}")
```
Output: `0.641514`. Matches **0.642** (Option C) exactly."""
    },
    # P13
    {
        "id": "P13",
        "title": "Multicollinearity and Variance Inflation Factor",
        "given": """- Multiple linear regression model
- Predictor: vehicle weight
- Variance Inflation Factor: $\text{VIF} = 14.8$""",
        "method": "Econometric properties of multicollinearity. VIF is defined as $\text{VIF}_j = \frac{1}{1 - R_j^2}$, measuring how much the variance of $\hat{\beta}_j$ is inflated due to linear correlation with other regressors.",
        "assumptions": "Gauss-Markov classical linear regression assumptions.",
        "computation": """Multicollinearity has the following specific properties:
1. **Unbiasedness**: OLS coefficient estimates remain unbiased and consistent ($E[\hat{\beta}] = \beta$). Hence Option A is false.
2. **Variance Inflation**: $\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{(n-1)s_j^2} \times \text{VIF}_j$. A VIF of 14.8 inflates standard errors by $\sqrt{14.8} \approx 3.85\times$, reducing t-statistics, widening confidence intervals, and degrading precision. Hence Option B is true.
3. Residual normality and linearity are separate assumptions evaluated by Q-Q plots and scatterplots/RESET tests, not VIF. Hence Options C and D are false.""",
        "conclusion": "Option B is the correct interpretation.",
        "audit": "Greene, Econometric Analysis (7th ed.), Ch. 4: Multicollinearity is a data-deficiency problem causing large variances and covariances of estimators, but does not violate Gauss-Markov conditions for unbiasedness. Option B verified."
    },
    # P14
    {
        "id": "P14",
        "title": "Heteroscedasticity from residual vs fitted plot",
        "given": """- OLS regression of wage equation
- Residual vs fitted plot shows a funnel/fan shape (dispersion increases with fitted values)""",
        "method": "Diagnostic identification of heteroscedasticity and Gauss-Markov theorem. A fan/funnel shape indicates non-constant error variance: $\text{Var}(\epsilon_i \mid X_i) = \sigma_i^2 \neq \sigma^2$.",
        "assumptions": "OLS regression diagnostics.",
        "computation": """Under heteroscedasticity:
1. OLS estimates remain linear and **unbiased** ($E[\hat{\beta}] = \beta$), because $E[\epsilon \mid X] = 0$ is not violated. Options A is false.
2. OLS standard errors $\sigma^2(X'X)^{-1}$ are **biased and inconsistent**, invalidating t-tests and F-tests. Option C is true.
3. Independence of errors refers to serial or spatial correlation ($E[\epsilon_i \epsilon_j] \neq 0$), which is distinct from non-constant variance. Option B is false.
4. Normality is separate from variance constancy. Option D is false.""",
        "conclusion": "Option C is the correct answer.",
        "audit": "Wooldridge, Introductory Econometrics, Ch. 8: 'Heteroskedasticity does not cause bias or inconsistency in OLS estimators... but OLS standard errors are no longer valid for hypothesis testing.' Option C verified."
    },
    # P15
    {
        "id": "P15",
        "title": "Durbin-Watson statistic and serial correlation",
        "given": """- Time series regression with $n = 80$ quarterly observations
- Residual Durbin-Watson statistic: $d = 0.58$""",
        "method": "Durbin-Watson test for first-order autoregressive serial correlation ($AR(1)$: $e_t = \rho e_{t-1} + u_t$). The statistic is defined as: $d = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2} \approx 2(1 - r)$, where $r$ is the sample autocorrelation of the residuals.",
        "assumptions": "Time-ordered observations with stationary AR(1) error process.",
        "computation": """The Durbin-Watson statistic ranges from 0 to 4:
- $d \approx 2 \implies r \approx 0$ (no autocorrelation)
- $d \to 0 \implies r \to +1$ (strong positive autocorrelation)
- $d \to 4 \implies r \to -1$ (strong negative autocorrelation)
Here, $d = 0.58 \approx 2(1 - r) \implies 1 - r \approx 0.29 \implies r \approx 0.71$.
Because $d = 0.58 \ll 2.0$, this indicates strong positive first-order autocorrelation.""",
        "conclusion": "Option A is the correct answer.",
        "audit": """**Method 2 (`statsmodels` formula verification)**:
$d \approx 2(1 - r) \implies r = 1 - d/2 = 1 - 0.58/2 = 1 - 0.29 = +0.71$.
A value of 0.58 is far below the lower critical bound $d_L \approx 1.5$ for $n=80$, decisively rejecting the null of zero correlation in favor of positive autocorrelation. Option A verified."""
    },
    # P16
    {
        "id": "P16",
        "title": "Q-Q plot interpretation of residual distribution",
        "given": """- Normal Q-Q plot of standardized residuals
- Shape: Plotted points deviate in an 'S' shape:
  - Lower tail: empirical quantiles are lower than theoretical normal quantiles.
  - Upper tail: empirical quantiles are higher than theoretical normal quantiles.""",
        "method": "Quantile-Quantile (Q-Q) Plot Diagnostic Interpretation. A Q-Q plot compares empirical order statistics to theoretical distribution quantiles.",
        "assumptions": "Comparison against standard normal theoretical quantiles.",
        "computation": """Let $q_{\text{emp}}$ be the empirical quantile and $q_{\text{theo}}$ be the theoretical quantile:
1. In the upper tail (positive $z$): $q_{\text{emp}} > q_{\text{theo}}$ means the largest positive residuals are more extreme (larger positive values) than normal.
2. In the lower tail (negative $z$): $q_{\text{emp}} < q_{\text{theo}}$ means the largest negative residuals are more extreme (more negative values) than normal.
3. When both tails exhibit more extreme values than the normal distribution, the distribution possesses **heavy tails** (leptokurtosis, excess kurtosis).
- Option A correctly identifies heavy tails / leptokurtosis.
- Option B (light tails) would exhibit the opposite curvature (empirical bounded within theoretical).
- Options C and D describe one-sided deviations (skewness).""",
        "conclusion": "Option A is the correct answer.",
        "audit": "Standard visualization theory (Cleveland 1993, Fox 2016): When both tails stretch beyond the 45-degree line ($y < x$ for $x < 0$ and $y > x$ for $x > 0$), the empirical distribution has heavier tails than the reference normal distribution. Option A verified."
    },
    # P17
    {
        "id": "P17",
        "title": "Bayes' theorem for rare disease screening",
        "given": """- Sensitivity: $P(T^+ \mid D) = 0.99$
- Specificity: $P(T^- \mid D^c) = 0.95 \implies P(T^+ \mid D^c) = 1 - 0.95 = 0.05$
- Prevalence (Prior): $P(D) = 0.001 \implies P(D^c) = 0.999$
- Observation: Single positive test $T^+$
- Rounding: 4 decimal places""",
        "method": "Bayes' Theorem for binary event: $P(D \mid T^+) = \frac{P(T^+ \mid D) P(D)}{P(T^+ \mid D) P(D) + P(T^+ \mid D^c) P(D^c)}$.",
        "assumptions": "Exhaustive binary partition $\{D, D^c\}$; known test performance characteristics.",
        "computation": """```python
p_d = 0.001
p_dc = 1 - p_d # 0.999
sens = 0.99
spec = 0.95
fpr = 1 - spec # 0.05

p_joint_true_pos = p_d * sens # 0.001 * 0.99 = 0.00099
p_joint_false_pos = p_dc * fpr # 0.999 * 0.05 = 0.04995
p_total_pos = p_joint_true_pos + p_joint_false_pos # 0.05094

posterior = p_joint_true_pos / p_total_pos # 0.00099 / 0.05094 = 99 / 5094 = 0.0194346...
```
Calculation details:
$$P(D \cap T^+) = 0.001 \times 0.99 = 0.00099$$
$$P(D^c \cap T^+) = 0.999 \times 0.05 = 0.04995$$
$$P(T^+) = 0.00099 + 0.04995 = 0.05094$$
$$P(D \mid T^+) = \frac{0.00099}{0.05094} = \frac{99}{5094} = \frac{33}{1698} \approx 0.0194346$$
Rounding to 4 decimal places gives **0.0194**.""",
        "conclusion": "Despite a 99% sensitive and 95% specific test, the posterior probability of disease given a positive test is only **0.0194** (1.94%), because false positives outnumber true positives by roughly 50 to 1.",
        "audit": """**Method 2 (Odds-Likelihood Formulation)**:
$$\text{Prior Odds} = \frac{0.001}{0.999} = \frac{1}{999}$$
$$\text{Bayes Factor} = \frac{\text{Sensitivity}}{1 - \text{Specificity}} = \frac{0.99}{0.05} = 19.8$$
$$\text{Posterior Odds} = \frac{1}{999} \times 19.8 = \frac{19.8}{999} = \frac{198}{9990} = \frac{11}{555} \approx 0.0198198$$
$$\text{Posterior Probability} = \frac{\text{Odds}}{1 + \text{Odds}} = \frac{11/555}{1 + 11/555} = \frac{11}{566} \approx 0.0194346$$
Both formulations yield $\frac{99}{5094} \approx 0.0194346$, rounding to **0.0194**."""
    },
    # P18
    {
        "id": "P18",
        "title": "Sequential Bayesian updating with two independent tests",
        "given": """- Prior: $P(D) = 0.001, P(D^c) = 0.999$
- Test 1 and Test 2 are conditionally independent given disease status.
- Each test has Sensitivity = 0.99 and Specificity = 0.95 ($FPR = 0.05$).
- Both tests return positive ($T_1^+, T_2^+$).
- Rounding: 4 decimal places""",
        "method": "Bayes' Theorem with conditional independence: $P(D \mid T_1^+, T_2^+) = \frac{P(D) P(T_1^+ \mid D) P(T_2^+ \mid D)}{P(D) P(T_1^+ \mid D) P(T_2^+ \mid D) + P(D^c) P(T_1^+ \mid D^c) P(T_2^+ \mid D^c)}$.",
        "assumptions": "Conditional independence of test outcomes given infection state.",
        "computation": """```python
p_d = 0.001
p_dc = 0.999
sens = 0.99
fpr = 0.05

p_joint_d = p_d * (sens ** 2) # 0.001 * 0.9801 = 0.0009801
p_joint_dc = p_dc * (fpr ** 2) # 0.999 * 0.0025 = 0.0024975
p_total = p_joint_d + p_joint_dc # 0.0034776

posterior = p_joint_d / p_total # 0.0009801 / 0.0034776 = 9801 / 34776 = 0.2818323...
```
Calculation details:
$$P(D \cap T_1^+ \cap T_2^+) = 0.001 \times (0.99)^2 = 0.001 \times 0.9801 = 0.0009801$$
$$P(D^c \cap T_1^+ \cap T_2^+) = 0.999 \times (0.05)^2 = 0.999 \times 0.0025 = 0.0024975$$
$$P(T_1^+ \cap T_2^+) = 0.0009801 + 0.0024975 = 0.0034776$$
$$P(D \mid T_1^+, T_2^+) = \frac{0.0009801}{0.0034776} = \frac{9801}{34776} \approx 0.281832$$
Rounding to 4 decimal places gives **0.2818**.""",
        "conclusion": "After two consecutive positive tests, the posterior probability of infection rises from 1.94% to **0.2818** (28.18%).",
        "audit": """**Method 2 (Sequential Updating with Intermediate Posterior as Prior)**:
Use the posterior of test 1 ($P_1 = \frac{99}{5094} \approx 0.0194346$) as the prior for test 2:
$$P(D \mid T_1^+) = \frac{99}{5094}, \quad P(D^c \mid T_1^+) = \frac{4995}{5094}$$
Numerator: $\frac{99}{5094} \times 0.99 = \frac{98.01}{5094}$
Denominator: $\frac{98.01}{5094} + \frac{4995}{5094} \times 0.05 = \frac{98.01 + 249.75}{5094} = \frac{347.76}{5094}$
Posterior: $\frac{98.01}{347.76} = \frac{9801}{34776} \approx 0.2818323$.
Matches **0.2818** exactly."""
    },
    # P19
    {
        "id": "P19",
        "title": "Multi-hypothesis Bayesian updating across three production lines",
        "given": """- Line 1: $P(M_1) = 0.50, P(\text{Defect} \mid M_1) = 0.02$
- Line 2: $P(M_2) = 0.30, P(\text{Defect} \mid M_2) = 0.04$
- Line 3: $P(M_3) = 0.20, P(\text{Defect} \mid M_3) = 0.10$
- Observation: Defective sensor
- Rounding: 4 decimal places""",
        "method": "Law of Total Probability and Bayes' Rule for partitions: $P(M_3 \mid \text{Defect}) = \frac{P(M_3) P(\text{Defect} \mid M_3)}{\sum_{i=1}^3 P(M_i) P(\text{Defect} \mid M_i)}$.",
        "assumptions": "Mutually exclusive and exhaustive production lines $\sum P(M_i) = 1.0$.",
        "computation": """```python
p_m = [0.50, 0.30, 0.20]
p_def_given_m = [0.02, 0.04, 0.10]

joints = [p * d for p, d in zip(p_m, p_def_given_m)]
# [0.010, 0.012, 0.020]

p_total_def = sum(joints) # 0.042
p_m3_given_def = joints[2] / p_total_def # 0.020 / 0.042 = 20 / 42 = 10 / 21 = 0.476190...
```
Calculation details:
$$P(M_1 \cap \text{Defect}) = 0.50 \times 0.02 = 0.010$$
$$P(M_2 \cap \text{Defect}) = 0.30 \times 0.04 = 0.012$$
$$P(M_3 \cap \text{Defect}) = 0.20 \times 0.10 = 0.020$$
$$P(\text{Defect}) = 0.010 + 0.012 + 0.020 = 0.042$$
$$P(M_3 \mid \text{Defect}) = \frac{0.020}{0.042} = \frac{20}{42} = \frac{10}{21} \approx 0.476190$$
Rounding to 4 decimal places gives **0.4762**.""",
        "conclusion": "The posterior probability that a defective sensor originated from Line $M_3$ is **0.4762** (47.62%).",
        "audit": """**Method 2 (Direct natural frequencies with base 1,000 sensors)**:
- Line 1: 500 sensors $\times 2\% = 10$ defectives.
- Line 2: 300 sensors $\times 4\% = 12$ defectives.
- Line 3: 200 sensors $\times 10\% = 20$ defectives.
- Total defectives = $10 + 12 + 20 = 42$.
- Defectives from Line 3 = 20.
- Ratio = $\frac{20}{42} = \frac{10}{21} \approx 0.476190$.
Matches **0.4762** identically."""
    },
    # P20
    {
        "id": "P20",
        "title": "Maximum A Posteriori (MAP) hypothesis selection",
        "given": """- Prior probabilities: $P(H_1) = 0.60, P(H_2) = 0.30, P(H_3) = 0.10$
- Evidence likelihoods: $P(E \mid H_1) = 0.15, P(E \mid H_2) = 0.40, P(E \mid H_3) = 0.80$
- 4 multiple-choice options (A: $H_1$, B: $H_2$, C: $H_3$, D: tie)""",
        "method": "Maximum A Posteriori (MAP) decision rule: $\hat{H}_{\text{MAP}} = \arg\max_{i} P(H_i \mid E) = \arg\max_{i} [P(H_i) P(E \mid H_i)]$, because the marginal evidence $P(E)$ is an identical positive constant for all hypotheses.",
        "assumptions": "Mutually exclusive and exhaustive hypotheses.",
        "computation": """```python
import numpy as np

priors = np.array([0.60, 0.30, 0.10])
likelihoods = np.array([0.15, 0.40, 0.80])
unnorm_posteriors = priors * likelihoods # [0.090, 0.120, 0.080]
p_evidence = np.sum(unnorm_posteriors) # 0.290
posteriors = unnorm_posteriors / p_evidence
# H1: 0.090 / 0.290 = 9/29 ≈ 0.3103 (31.03%)
# H2: 0.120 / 0.290 = 12/29 ≈ 0.4138 (41.38%)
# H3: 0.080 / 0.290 = 8/29 ≈ 0.2759 (27.59%)
```
Joint calculation:
- For $H_1$: $0.60 \times 0.15 = 0.090$
- For $H_2$: $0.30 \times 0.40 = 0.120$
- For $H_3$: $0.10 \times 0.80 = 0.080$
Comparing joint probabilities:
$$0.120 > 0.090 > 0.080 \implies P(H_2 \mid E) > P(H_1 \mid E) > P(H_3 \mid E)$$
$H_2$ achieves the highest posterior probability ($41.38\%$), which corresponds to Option B.""",
        "conclusion": "Hypothesis $H_2$ has the highest posterior probability (Option B).",
        "audit": """**Method 2 (Pairwise Odds Updates)**:
Prior odds $H_2$ vs $H_1$: $\frac{0.30}{0.60} = 0.5$. Bayes factor: $\frac{0.40}{0.15} = \frac{8}{3} \approx 2.6667$.
Posterior odds $H_2$ vs $H_1$: $0.5 \times \frac{8}{3} = \frac{4}{3} \approx 1.333 > 1 \implies H_2$ beats $H_1$.
Prior odds $H_2$ vs $H_3$: $\frac{0.30}{0.10} = 3.0$. Bayes factor: $\frac{0.40}{0.80} = 0.5$.
Posterior odds $H_2$ vs $H_3$: $3.0 \times 0.5 = 1.5 > 1 \implies H_2$ beats $H_3$.
Since $H_2$ beats both $H_1$ and $H_3$, $H_2$ is confirmed as the MAP hypothesis (Option B)."""
    }
]

for ps in problem_solutions:
    solutions_md += f"""
## Problem {ps['id']}: {ps['title']}

### 1. Given Information
{ps['given']}

### 2. Method Choice and Justification
{ps['method']}

### 3. Assumption Checks
{ps['assumptions']}

### 4. Step-by-Step Computation & Code
{ps['computation']}

### 5. Conclusion in Context
{ps['conclusion']}

### 6. Self-Audit & Dual Derivation
{ps['audit']}

---
"""

with open("solutions.md", "w", encoding="utf-8") as f:
    f.write(solutions_md)
print("Saved solutions.md successfully.")
