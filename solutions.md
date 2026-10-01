# Statistics Benchmark Reference Solutions & Dual-Method Self-Audit

This document contains the official reference solutions for all 20 problems in the statistics benchmark.
Every problem follows a strict 5-part structure:
1. **Given Information**
2. **Method Choice & Justification**
3. **Assumption Checks**
4. **Step-by-Step Computation & Python Verification**
5. **Conclusion in Context**
6. **Self-Audit / Dual-Method Re-Derivation**

---

## Problem P01: One-sample z-test for a population proportion

### 1. Given Information
- Null hypothesis: $H_0: p \le 0.05$ (or $p = 0.05$)
- Alternative hypothesis: $H_1: p > 0.05$ (one-tailed, upper-tail test)
- Sample size: $n = 400$
- Number of defectives: $x = 28$
- Sample proportion: $\hat{p} = rac{28}{400} = 0.07$
- Significance level: $lpha = 0.05$
- Standard normal approximation without continuity correction

### 2. Method Choice and Justification
One-sample z-test for proportions using the null proportion $p_0$ to compute standard error (Score test). In hypothesis testing for a single proportion, standard error is evaluated under the null hypothesis: $SE_0 = \sqrt{rac{p_0(1-p_0)}{n}}$.

### 3. Assumption Checks
1. **Random sampling**: The 400 microchips constitute a simple random sample.
2. **Independence**: $n = 400$ is much less than 10% of total plant production ($10\% 	ext{ condition}$).
3. **Success/Failure condition**: $n p_0 = 400 	imes 0.05 = 20 \ge 10$ and $n(1 - p_0) = 400 	imes 0.95 = 380 \ge 10$. Both conditions are satisfied, justifying normal approximation.

### 4. Step-by-Step Computation & Code
```python
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
$$SE_0 = \sqrt{rac{0.05 	imes 0.95}{400}} = \sqrt{rac{0.0475}{400}} = \sqrt{0.00011875} pprox 0.01089725$$
$$z = rac{0.07 - 0.05}{0.01089725} = rac{0.02}{0.01089725} pprox 1.83533$$
Rounding to 3 decimal places gives **1.835**.

### 5. Conclusion in Context
The test statistic is $z = 1.835$. Because $z = 1.835 > z_{0.05} = 1.645$ (corresponding to $p = 0.0332 < 0.05$), we reject the null hypothesis at $lpha = 0.05$ and conclude there is sufficient evidence that the defect rate exceeds 5%.

### 6. Self-Audit & Dual Derivation
**Method 2 (Statsmodels `proportions_ztest` / Exact Binomial Check)**:
```python
from statsmodels.stats.proportion import proportions_ztest, binom_test
# Score z-test:
z_sm, p_sm = proportions_ztest(count=28, nobs=400, value=0.05, alternative='larger', prop_var=0.05)
print(f"Statsmodels z: {z_sm:.3f}, p-value: {p_sm:.4f}")
# Exact Binomial:
p_exact = stats.binomtest(28, 400, 0.05, alternative='greater').pvalue
print(f"Exact Binomial p-value: {p_exact:.4f}")
```
Output: `z = 1.835`, `p = 0.0332`. Both analytic and statsmodels evaluations match exactly at **1.835**.

---

## Problem P02: Paired t-test for dependent samples

### 1. Given Information
- Sample size: $n = 10$ patients
- Paired differences ($d_i = 	ext{After}_i - 	ext{Before}_i$): $[-6, -4, -8, -5, -7, -3, -9, -4, -6, -8]$
- Null hypothesis: $H_0: \mu_d = 0$
- Alternative hypothesis: $H_1: \mu_d 
eq 0$ (two-tailed)
- Significance level: $lpha = 0.01$
- Population differences are approximately normally distributed.

### 2. Method Choice and Justification
Paired-samples Student's t-test. The measurements are repeated before-and-after observations on the same 10 individuals, creating within-subject pairing. The test reduces to a one-sample t-test on the differences $d_i$: $t = rac{ar{d} - 0}{s_d / \sqrt{n}}$ with $df = n - 1 = 9$.

### 3. Assumption Checks
1. **Paired observations**: Each pair of measurements corresponds to the same subject.
2. **Independence across subjects**: Patient responses are independent across the sample.
3. **Normality of differences**: The distribution of differences $d_i$ is stated to be approximately normal.

### 4. Step-by-Step Computation & Code
```python
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
$$\sum d_i = -60 \implies ar{d} = -6.0$$
$$\sum (d_i - ar{d})^2 = 0 + 4 + 4 + 1 + 1 + 9 + 9 + 4 + 0 + 4 = 36 \implies s_d = \sqrt{rac{36}{9}} = \sqrt{4} = 2.0$$
$$SE = rac{2.0}{\sqrt{10}} pprox 0.6324555$$
$$t = rac{-6.0}{0.6324555} = -3\sqrt{10} pprox -9.48683$$
Rounding to 3 decimal places gives **-9.487**.

### 5. Conclusion in Context
The test statistic is $t = -9.487$ ($df = 9, p = 5.64 	imes 10^{-6}$). Since $|t| = 9.487 > t_{0.005, 9} = 3.250$, we reject $H_0$ at $lpha = 0.01$ and conclude the drug causes a statistically significant reduction in systolic blood pressure.

### 6. Self-Audit & Dual Derivation
**Method 2 (`scipy.stats.ttest_1samp` / manual sum of squares)**:
```python
t_check, p_check = stats.ttest_1samp(d, 0)
print(f"Direct t-test: t = {t_check:.4f}, exact value = {-3*np.sqrt(10):.4f}")
```
Both yield $t = -3\sqrt{10} pprox -9.48683$, rounding to **-9.487**.

---

## Problem P03: Welch's two-sample t-test with unequal variances

### 1. Given Information
- Sample 1: $n_1 = 12, ar{x}_1 = 24.5, s_1 = 1.8$
- Sample 2: $n_2 = 25, ar{x}_2 = 21.0, s_2 = 4.2$
- Population variances: $\sigma_1^2 
eq \sigma_2^2$ (unequal / unpooled)
- Hypotheses: $H_0: \mu_1 - \mu_2 = 0$ vs $H_1: \mu_1 - \mu_2 
eq 0$
- Rounding: 3 decimal places

### 2. Method Choice and Justification
Welch's t-test (unequal variances t-test). When population variances cannot be assumed equal ($s_2^2 / s_1^2 = 4.2^2 / 1.8^2 = 17.64 / 3.24 pprox 5.44$), pooling is inappropriate. The standard error is: $SE = \sqrt{rac{s_1^2}{n_1} + rac{s_2^2}{n_2}}$, and the test statistic is $t = rac{ar{x}_1 - ar{x}_2}{SE}$.

### 3. Assumption Checks
1. **Independent random samples**: Two distinct plots randomized to soil treatments.
2. **Normality**: Crop yields within plots are normally distributed.
3. **Heteroscedasticity**: $\sigma_1^2 
eq \sigma_2^2$, requiring Welch's formulation without pooling.

### 4. Step-by-Step Computation & Code
```python
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
$$rac{s_1^2}{n_1} = rac{1.8^2}{12} = rac{3.24}{12} = 0.2700$$
$$rac{s_2^2}{n_2} = rac{4.2^2}{25} = rac{17.64}{25} = 0.7056$$
$$SE = \sqrt{0.2700 + 0.7056} = \sqrt{0.9756} pprox 0.9877247$$
$$t = rac{24.5 - 21.0}{0.9877247} = rac{3.5}{0.9877247} pprox 3.54350$$
Rounding to 3 decimal places gives **3.543**.

### 5. Conclusion in Context
The Welch t-statistic is $t = 3.543$ ($df = 34.77, p = 0.0011$). At any standard significance level ($lpha = 0.05$ or $0.01$), we reject the null hypothesis and conclude that the soil treatments produce significantly different mean crop yields.

### 6. Self-Audit & Dual Derivation
**Method 2 (`scipy.stats.ttest_ind_from_stats` with `equal_var=False`)**:
```python
t_check, p_check = stats.ttest_ind_from_stats(
    mean1=24.5, std1=1.8, nobs1=12,
    mean2=21.0, std2=4.2, nobs2=25,
    equal_var=False
)
print(f"Scipy Welch t: {t_check:.4f}, p: {p_check:.6f}")
```
Output: `t = 3.5435`, `p = 0.001149`. Matches **3.543** perfectly.

---

## Problem P04: Pearson's Chi-square test of independence

### 1. Given Information
- $2 	imes 2$ Contingency Table:
  - Row 1 (Treatment A): 40 Recovered, 60 Not Recovered (Total = 100)
  - Row 2 (Treatment B): 60 Recovered, 40 Not Recovered (Total = 100)
- Column totals: Recovered = 100, Not Recovered = 100
- Grand Total: $N = 200$
- No Yates' continuity correction.

### 2. Method Choice and Justification
Pearson's Chi-Square Test of Independence without continuity correction. For an $r 	imes c$ table, expected cell frequency is $E_{ij} = rac{R_i C_j}{N}$, and test statistic is $\chi^2 = \sum rac{(O_{ij} - E_{ij})^2}{E_{ij}}$ with $df = (r-1)(c-1) = 1$.

### 3. Assumption Checks
1. **Categorical data**: Both treatment and recovery are binary categorical variables.
2. **Independent observations**: Each patient is counted in exactly one cell.
3. **Expected frequencies**: All expected cell counts $E_{ij} = 50 \ge 5$, easily meeting Cochran's criterion.

### 4. Step-by-Step Computation & Code
```python
import numpy as np
import scipy.stats as stats

obs = np.array([[40, 60], [60, 40]])
chi2, p, dof, ex = stats.chi2_contingency(obs, correction=False)
```
Calculation details:
$$E_{11} = rac{100 	imes 100}{200} = 50, \quad E_{12} = 50, \quad E_{21} = 50, \quad E_{22} = 50$$
$$\chi^2 = rac{(40 - 50)^2}{50} + rac{(60 - 50)^2}{50} + rac{(60 - 50)^2}{50} + rac{(40 - 50)^2}{50}$$
$$\chi^2 = rac{100}{50} + rac{100}{50} + rac{100}{50} + rac{100}{50} = 2 + 2 + 2 + 2 = 8.000$$
Rounding to 3 decimal places gives **8.000**.

### 5. Conclusion in Context
The test statistic is $\chi^2 = 8.000$ ($df = 1, p = 0.0047$). Since $\chi^2 = 8.000 > \chi^2_{0.05, 1} = 3.841$, we reject independence and conclude recovery rates differ significantly between treatments.

### 6. Self-Audit & Dual Derivation
**Method 2 (Two-proportion z-test relationship $\chi^2 = z^2$)**:
For a $2 	imes 2$ table, Pearson $\chi^2$ equals the squared pooled two-proportion z-statistic:
$$\hat{p}_1 = 0.40, \hat{p}_2 = 0.60, \hat{p}_{	ext{pool}} = rac{40+60}{200} = 0.50$$
$$SE_{	ext{pool}} = \sqrt{0.5 	imes 0.5 	imes (1/100 + 1/100)} = \sqrt{0.25 	imes 0.02} = \sqrt{0.005} pprox 0.07071068$$
$$z = rac{0.40 - 0.60}{0.07071068} = rac{-0.20}{0.07071068} = -\sqrt{8} pprox -2.828427$$
$$z^2 = (-\sqrt{8})^2 = 8.000$$
Both methods match identically at **8.000**.

---

## Problem P05: Confidence interval for normal mean with small n and unknown sigma

### 1. Given Information
- Sample size: $n = 9$
- Sample mean: $ar{x} = 50.0 	ext{ mg/L}$
- Sample standard deviation: $s = 6.0 	ext{ mg/L}$
- Population: Normally distributed with unknown variance $\sigma^2$
- Confidence level: $1 - lpha = 0.95 \implies lpha = 0.05$, two-sided

### 2. Method Choice and Justification
Student's t confidence interval. Because the population standard deviation $\sigma$ is unknown and estimated by sample standard deviation $s$ with a small sample size ($n = 9$), the pivot $rac{ar{x} - \mu}{s/\sqrt{n}}$ follows Student's t distribution with $df = n - 1 = 8$.

### 3. Assumption Checks
1. **Normality**: The underlying population is normally distributed.
2. **Random sampling**: Batches are produced independently under identical conditions.
3. **Unknown variance**: $\sigma$ is unknown, requiring Student's t critical value rather than normal $z$.

### 4. Step-by-Step Computation & Code
```python
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
$$SE = rac{6.0}{\sqrt{9}} = rac{6.0}{3} = 2.0$$
$$t_{0.025, 8} = 2.3060041$$
$$E = 2.3060041 	imes 2.0 = 4.612008$$
$$	ext{Upper Limit} = 50.0 + 4.612008 pprox 54.612$$
Rounding to 3 decimal places gives **54.612**.

### 5. Conclusion in Context
We are 95% confident that the true population mean active compound concentration is between $45.388 	ext{ mg/L}$ and $54.612 	ext{ mg/L}$. The requested upper limit is **54.612**.

### 6. Self-Audit & Dual Derivation
**Method 2 (`scipy.stats.t.interval`)**:
```python
ci_low, ci_high = stats.t.interval(0.95, df=8, loc=50.0, scale=6.0/np.sqrt(9))
print(f"Scipy interval: [{ci_low:.4f}, {ci_high:.4f}]")
```
Output: `ci_high = 54.6120`. Matches **54.612** exactly.

---

## Problem P06: Confidence interval for difference in means with known population variances

### 1. Given Information
- Sample 1: $n_1 = 15, ar{x}_1 = 78.0, \sigma_1^2 = 16.0 \implies \sigma_1 = 4.0$
- Sample 2: $n_2 = 20, ar{x}_2 = 72.0, \sigma_2^2 = 25.0 \implies \sigma_2 = 5.0$
- Population variances are KNOWN
- Confidence level: $99\% \implies lpha = 0.01$, two-sided $lpha/2 = 0.005$

### 2. Method Choice and Justification
Standard normal (z-distribution) confidence interval for two independent means with known population variances. Because $\sigma_1^2$ and $\sigma_2^2$ are known parameters, the exact sampling distribution of $rac{(ar{x}_1 - ar{x}_2) - (\mu_1 - \mu_2)}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}}$ is standard normal $\mathcal{N}(0, 1)$, irrespective of small sample size.

### 3. Assumption Checks
1. **Known variances**: Population variances $\sigma_1^2 = 16$ and $\sigma_2^2 = 25$ are true parameters, not sample estimates.
2. **Normality**: Both fill volume populations are normally distributed.
3. **Independence**: Independent random samples between machines.

### 4. Step-by-Step Computation & Code
```python
import numpy as np
import scipy.stats as stats

sig1_sq, n1 = 16.0, 15
sig2_sq, n2 = 25.0, 20
z_crit = stats.norm.ppf(0.995) # 2.5758293...
se = np.sqrt(sig1_sq / n1 + sig2_sq / n2) # sqrt(16/15 + 25/20) = sqrt(1.066667 + 1.25) = sqrt(2.316667) = 1.5220599...
margin_of_error = z_crit * se # 2.5758293 * 1.5220599 = 3.920566...
```
Calculation details:
$$SE = \sqrt{rac{16}{15} + rac{25}{20}} = \sqrt{rac{16}{15} + rac{5}{4}} = \sqrt{rac{64 + 75}{60}} = \sqrt{rac{139}{60}} pprox 1.5220599$$
$$z_{0.005} pprox 2.5758293$$
$$E = 2.5758293 	imes 1.5220599 pprox 3.92057$$
Rounding to 3 decimal places gives **3.921**.

### 5. Conclusion in Context
The margin of error for a 99% confidence interval of $\mu_1 - \mu_2$ is **3.921** mL.

### 6. Self-Audit & Dual Derivation
**Method 2 (`scipy.stats.norm.interval`)**:
```python
se_val = np.sqrt(139.0 / 60.0)
ci = stats.norm.interval(0.99, loc=0, scale=se_val)
me_scipy = ci[1]
print(f"Scipy norm margin of error: {me_scipy:.4f}")
```
Output: `me_scipy = 3.9206`, rounding to **3.921**.

---

## Problem P07: Wald confidence interval lower bound for a single proportion

### 1. Given Information
- Sample size: $n = 100$
- Number of successes: $x = 35$
- Sample proportion: $\hat{p} = rac{35}{100} = 0.35$
- Confidence level: $95\% \implies lpha = 0.05, z_{0.025} pprox 1.959964$
- Standard Wald confidence interval formula specified.

### 2. Method Choice and Justification
Standard Wald confidence interval for a single proportion: $\hat{p} \pm z_{lpha/2} \sqrt{rac{\hat{p}(1-\hat{p})}{n}}$.

### 3. Assumption Checks
1. **Random sampling**: Survey respondents represent a random sample of voters.
2. **Success/Failure condition**: $n \hat{p} = 35 \ge 10$ and $n(1 - \hat{p}) = 65 \ge 10$, satisfying the empirical rule of thumb for asymptotic normality.

### 4. Step-by-Step Computation & Code
```python
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
$$SE = \sqrt{rac{0.35 	imes 0.65}{100}} = \sqrt{0.002275} pprox 0.04769696$$
$$E = 1.959964 	imes 0.04769696 pprox 0.0934843$$
$$	ext{Lower Bound} = 0.35 - 0.0934843 pprox 0.256516$$
Rounding to 4 decimal places gives **0.2565**.

### 5. Conclusion in Context
The lower bound of the standard 95% Wald confidence interval for voter support is **0.2565** (25.65%).

### 6. Self-Audit & Dual Derivation
**Method 2 (`statsmodels.stats.proportion.proportion_confint` with `method='normal'`)**:
```python
from statsmodels.stats.proportion import proportion_confint
ci_low, ci_high = proportion_confint(count=35, nobs=100, alpha=0.05, method='normal')
print(f"Statsmodels Wald CI: [{ci_low:.6f}, {ci_high:.6f}]")
```
Output: `ci_low = 0.256516`. Matches **0.2565** exactly.

---

## Problem P08: Confidence interval margin of error for difference in proportions

### 1. Given Information
- Design A: $n_1 = 200, x_1 = 80 \implies \hat{p}_1 = rac{80}{200} = 0.40$
- Design B: $n_2 = 250, x_2 = 60 \implies \hat{p}_2 = rac{60}{250} = 0.24$
- Confidence level: $90\% \implies lpha = 0.10, lpha/2 = 0.05, z_{0.05} pprox 1.6448536$
- Unpooled Wald method for independent samples specified.

### 2. Method Choice and Justification
Unpooled Wald confidence interval for the difference of two independent proportions: $E = z_{lpha/2} \sqrt{rac{\hat{p}_1(1-\hat{p}_1)}{n_1} + rac{\hat{p}_2(1-\hat{p}_2)}{n_2}}$.

### 3. Assumption Checks
1. **Independent samples**: Visitors to Design A and Design B are independently randomized.
2. **Success/Failure condition**: $n_1 \hat{p}_1 = 80 \ge 10$, $n_1(1-\hat{p}_1) = 120 \ge 10$, $n_2 \hat{p}_2 = 60 \ge 10$, $n_2(1-\hat{p}_2) = 190 \ge 10$.
3. **No pooling**: In confidence interval estimation, proportions are unpooled because the true difference is not hypothesized to be zero.

### 4. Step-by-Step Computation & Code
```python
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
$$rac{\hat{p}_1(1-\hat{p}_1)}{n_1} = rac{0.40 	imes 0.60}{200} = rac{0.24}{200} = 0.0012000$$
$$rac{\hat{p}_2(1-\hat{p}_2)}{n_2} = rac{0.24 	imes 0.76}{250} = rac{0.1824}{250} = 0.0007296$$
$$SE = \sqrt{0.0012000 + 0.0007296} = \sqrt{0.0019296} pprox 0.0439272$$
$$E = 1.6448536 	imes 0.0439272 pprox 0.0722538$$
Rounding to 4 decimal places gives **0.0723**.

### 5. Conclusion in Context
The margin of error for a 90% confidence interval of the conversion rate difference $(p_1 - p_2)$ is **0.0723** (7.23 percentage points).

### 6. Self-Audit & Dual Derivation
**Method 2 (`statsmodels.stats.proportion.confint_proportions_2indep`)**:
```python
from statsmodels.stats.proportion import confint_proportions_2indep
ci_low, ci_high = confint_proportions_2indep(80, 200, 60, 250, method='wald', alpha=0.10)
diff = (80/200) - (60/250) # 0.16
me_sm = (ci_high - ci_low) / 2.0
print(f"Statsmodels unpooled ME: {me_sm:.6f}")
```
Output: `me_sm = 0.072254`. Matches **0.0723** exactly.

---

## Problem P09: Formal definition of a p-value

### 1. Given Information
- Scenario: Medical trial reports two-tailed $p = 0.03$.
- 4 multiple-choice statements defining the p-value.

### 2. Method Choice and Justification
Frequentist statistical definition analysis. By definition, a p-value is the probability, calculated under the assumption that the null hypothesis $H_0$ is true, of observing a test statistic as extreme as or more extreme than the observed sample outcome.

### 3. Assumption Checks
Frequentist null hypothesis significance testing framework.

### 4. Step-by-Step Computation & Code
Distractor Analysis:
- Option A: "There is a 3% probability that the null hypothesis is true." -> False. This commits the classic inverse probability fallacy: $P(H_0 \mid D) 
eq P(	ext{data} \mid H_0)$. Hypotheses are fixed parameters in frequentist statistics, not random variables with probabilities.
- Option B: "There is a 97% probability that the new medication is superior." -> False. Bayesian posterior probability cannot be inferred from a p-value without a prior distribution.
- Option C: "Assuming that the null hypothesis is true, the probability of observing a test statistic at least as extreme as the one calculated from the sample data is 0.03." -> Exactly the textbook mathematical definition of a p-value: $P(T \ge t_{	ext{obs}} \mid H_0)$.
- Option D: "The probability of obtaining a false positive result if this experiment is replicated is 0.03." -> False. Replicability involves power and true effect size, not p-value.

### 5. Conclusion in Context
Option C is the only mathematically and methodologically correct definition.

### 6. Self-Audit & Dual Derivation
ASA Statement on Statistical Significance and P-Values (Wasserstein & Lazar, 2016): 'Informally, a p-value is the probability under a specified statistical model that a statistical summary of the data would be equal to or more extreme than its observed value.' Option C is verified as the unique correct answer.

---

## Problem P10: Statistical significance versus practical significance

### 1. Given Information
- Sample size: $n = 500,000$ office workers
- Measured effect: reduction in sedentary time of 42 seconds (95% CI: [38, 46] seconds)
- P-value: $p < 0.0001$
- Executive claim: "overwhelming practical health benefits and transforms workplace activity levels".

### 2. Method Choice and Justification
Statistical vs Practical Significance Analysis. The standard error of an estimate scales as $\sigma / \sqrt{n}$. As $n 	o \infty$, $SE 	o 0$, causing any nonzero effect, however microscopic or practically meaningless, to yield an infinitesimal p-value.

### 3. Assumption Checks
Large sample properties of test statistics.

### 4. Step-by-Step Computation & Code
An average change of 42 seconds in a workday of 28,800 seconds (8 hours) represents a relative change of $42 / 28800 pprox 0.15\%$. While the narrow CI [38, 46] establishes that the 42-second reduction is precisely estimated and not a sampling artifact ($p < 0.0001$), 42 seconds is practically inconsequential for health outcomes.
- Option A asserts small p-values guarantee large effect magnitude (false).
- Option B correctly identifies that in huge samples, trivial effect sizes yield tiny p-values (true).
- Option C confuses estimation precision with clinical magnitude (false).
- Option D makes a bogus assertion about Gauss-Markov sample size limits (false).

### 5. Conclusion in Context
Option B is the correct evaluation.

### 6. Self-Audit & Dual Derivation
Well-established in statistical literature (e.g., Lin et al., 2013 'Too Big to Fail: Large Samples and the p-Value Problem'). With $n = 500,000$, statistical power to detect trivial effect sizes approaches 1.0, rendering p-values uninformative about practical importance. Option B verified.

---

## Problem P11: Absence of evidence versus evidence of absence

### 1. Given Information
- Sample size: $n = 12$ patients
- Test: two-tailed paired t-test
- Result: $t = 0.91, p = 0.38$
- Investigator conclusion: "Because $p > 0.05$, we accept the null hypothesis and conclude the compound produces zero arrhythmic side effects." 

### 2. Method Choice and Justification
Logic of Hypothesis Testing & Statistical Power. In frequentist hypothesis testing, failing to reject $H_0$ means data do not provide sufficient evidence against $H_0$; it never proves $H_0$ is true ('Absence of evidence is not evidence of absence').

### 3. Assumption Checks
Statistical power with small sample size ($n=12$).

### 4. Step-by-Step Computation & Code
With $n = 12$, a two-sample or paired t-test has very low statistical power to detect small or moderate effect sizes. A high p-value ($p = 0.38$) could easily occur even if the compound has real, dangerous side effects.
- Option A correctly states that failing to reject does not prove $H_0$ and points to low power from small sample size.
- Option B commits the inverse probability error ($P(H_1) = 0.38$).
- Option C is fabricated nonsense.
- Option D confuses one-tailed conversions.

### 5. Conclusion in Context
Option A is the unique methodologically correct explanation.

### 6. Self-Audit & Dual Derivation
Standard statistical dogma (Altman & Bland 1995: 'Absence of evidence is not evidence of absence'). Equivalence testing (TOST) would be required to claim equivalence or zero effect, not failure to reject in a small underpowered sample. Option A verified.

---

## Problem P12: Multiple testing and family-wise error rate

### 1. Given Information
- Number of independent tests: $m = 20$
- Nominal per-comparison error rate: $lpha = 0.05$
- True state: all 20 null hypotheses are true ($H_{0,1}, \dots, H_{0,20}$ are true)
- No multiple testing adjustment applied.

### 2. Method Choice and Justification
Family-Wise Error Rate (FWER) probability calculation under independence. For $m$ independent true null hypotheses, the probability of at least one false positive is: $	ext{FWER} = 1 - (1 - lpha)^m$.

### 3. Assumption Checks
Independence across all 20 tests; all null hypotheses true.

### 4. Step-by-Step Computation & Code
```python
alpha = 0.05
m = 20
p_no_type1 = (1 - alpha)**m # 0.95^20 = 0.3584859...
fwer = 1 - p_no_type1 # 1 - 0.3584859 = 0.641514...
```
Calculation details:
$$P(	ext{No Type I errors}) = (1 - 0.05)^{20} = (0.95)^{20} pprox 0.358486$$
$$	ext{FWER} = 1 - P(	ext{No Type I errors}) = 1 - 0.358486 pprox 0.641514$$
Rounding to 3 decimal places gives **0.642**, which corresponds to Option C.

### 5. Conclusion in Context
The probability of at least one false positive is 0.642 (Option C).

### 6. Self-Audit & Dual Derivation
**Method 2 (`scipy.stats.binom.sf`)**:
Using the Binomial survival function $P(X \ge 1)$ where $X \sim 	ext{Binomial}(20, 0.05)$:
```python
p_binom = stats.binom.sf(0, 20, 0.05)
print(f"Binomial SF: {p_binom:.6f}")
```
Output: `0.641514`. Matches **0.642** (Option C) exactly.

---

## Problem P13: Multicollinearity and Variance Inflation Factor

### 1. Given Information
- Multiple linear regression model
- Predictor: vehicle weight
- Variance Inflation Factor: $	ext{VIF} = 14.8$

### 2. Method Choice and Justification
Econometric properties of multicollinearity. VIF is defined as $	ext{VIF}_j = rac{1}{1 - R_j^2}$, measuring how much the variance of $\hat{eta}_j$ is inflated due to linear correlation with other regressors.

### 3. Assumption Checks
Gauss-Markov classical linear regression assumptions.

### 4. Step-by-Step Computation & Code
Multicollinearity has the following specific properties:
1. **Unbiasedness**: OLS coefficient estimates remain unbiased and consistent ($E[\hat{eta}] = eta$). Hence Option A is false.
2. **Variance Inflation**: $	ext{Var}(\hat{eta}_j) = rac{\sigma^2}{(n-1)s_j^2} 	imes 	ext{VIF}_j$. A VIF of 14.8 inflates standard errors by $\sqrt{14.8} pprox 3.85	imes$, reducing t-statistics, widening confidence intervals, and degrading precision. Hence Option B is true.
3. Residual normality and linearity are separate assumptions evaluated by Q-Q plots and scatterplots/RESET tests, not VIF. Hence Options C and D are false.

### 5. Conclusion in Context
Option B is the correct interpretation.

### 6. Self-Audit & Dual Derivation
Greene, Econometric Analysis (7th ed.), Ch. 4: Multicollinearity is a data-deficiency problem causing large variances and covariances of estimators, but does not violate Gauss-Markov conditions for unbiasedness. Option B verified.

---

## Problem P14: Heteroscedasticity from residual vs fitted plot

### 1. Given Information
- OLS regression of wage equation
- Residual vs fitted plot shows a funnel/fan shape (dispersion increases with fitted values)

### 2. Method Choice and Justification
Diagnostic identification of heteroscedasticity and Gauss-Markov theorem. A fan/funnel shape indicates non-constant error variance: $	ext{Var}(\epsilon_i \mid X_i) = \sigma_i^2 
eq \sigma^2$.

### 3. Assumption Checks
OLS regression diagnostics.

### 4. Step-by-Step Computation & Code
Under heteroscedasticity:
1. OLS estimates remain linear and **unbiased** ($E[\hat{eta}] = eta$), because $E[\epsilon \mid X] = 0$ is not violated. Options A is false.
2. OLS standard errors $\sigma^2(X'X)^{-1}$ are **biased and inconsistent**, invalidating t-tests and F-tests. Option C is true.
3. Independence of errors refers to serial or spatial correlation ($E[\epsilon_i \epsilon_j] 
eq 0$), which is distinct from non-constant variance. Option B is false.
4. Normality is separate from variance constancy. Option D is false.

### 5. Conclusion in Context
Option C is the correct answer.

### 6. Self-Audit & Dual Derivation
Wooldridge, Introductory Econometrics, Ch. 8: 'Heteroskedasticity does not cause bias or inconsistency in OLS estimators... but OLS standard errors are no longer valid for hypothesis testing.' Option C verified.

---

## Problem P15: Durbin-Watson statistic and serial correlation

### 1. Given Information
- Time series regression with $n = 80$ quarterly observations
- Residual Durbin-Watson statistic: $d = 0.58$

### 2. Method Choice and Justification
Durbin-Watson test for first-order autoregressive serial correlation ($AR(1)$: $e_t = ho e_{t-1} + u_t$). The statistic is defined as: $d = rac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2} pprox 2(1 - r)$, where $r$ is the sample autocorrelation of the residuals.

### 3. Assumption Checks
Time-ordered observations with stationary AR(1) error process.

### 4. Step-by-Step Computation & Code
The Durbin-Watson statistic ranges from 0 to 4:
- $d pprox 2 \implies r pprox 0$ (no autocorrelation)
- $d 	o 0 \implies r 	o +1$ (strong positive autocorrelation)
- $d 	o 4 \implies r 	o -1$ (strong negative autocorrelation)
Here, $d = 0.58 pprox 2(1 - r) \implies 1 - r pprox 0.29 \implies r pprox 0.71$.
Because $d = 0.58 \ll 2.0$, this indicates strong positive first-order autocorrelation.

### 5. Conclusion in Context
Option A is the correct answer.

### 6. Self-Audit & Dual Derivation
**Method 2 (`statsmodels` formula verification)**:
$d pprox 2(1 - r) \implies r = 1 - d/2 = 1 - 0.58/2 = 1 - 0.29 = +0.71$.
A value of 0.58 is far below the lower critical bound $d_L pprox 1.5$ for $n=80$, decisively rejecting the null of zero correlation in favor of positive autocorrelation. Option A verified.

---

## Problem P16: Q-Q plot interpretation of residual distribution

### 1. Given Information
- Normal Q-Q plot of standardized residuals
- Shape: Plotted points deviate in an 'S' shape:
  - Lower tail: empirical quantiles are lower than theoretical normal quantiles.
  - Upper tail: empirical quantiles are higher than theoretical normal quantiles.

### 2. Method Choice and Justification
Quantile-Quantile (Q-Q) Plot Diagnostic Interpretation. A Q-Q plot compares empirical order statistics to theoretical distribution quantiles.

### 3. Assumption Checks
Comparison against standard normal theoretical quantiles.

### 4. Step-by-Step Computation & Code
Let $q_{	ext{emp}}$ be the empirical quantile and $q_{	ext{theo}}$ be the theoretical quantile:
1. In the upper tail (positive $z$): $q_{	ext{emp}} > q_{	ext{theo}}$ means the largest positive residuals are more extreme (larger positive values) than normal.
2. In the lower tail (negative $z$): $q_{	ext{emp}} < q_{	ext{theo}}$ means the largest negative residuals are more extreme (more negative values) than normal.
3. When both tails exhibit more extreme values than the normal distribution, the distribution possesses **heavy tails** (leptokurtosis, excess kurtosis).
- Option A correctly identifies heavy tails / leptokurtosis.
- Option B (light tails) would exhibit the opposite curvature (empirical bounded within theoretical).
- Options C and D describe one-sided deviations (skewness).

### 5. Conclusion in Context
Option A is the correct answer.

### 6. Self-Audit & Dual Derivation
Standard visualization theory (Cleveland 1993, Fox 2016): When both tails stretch beyond the 45-degree line ($y < x$ for $x < 0$ and $y > x$ for $x > 0$), the empirical distribution has heavier tails than the reference normal distribution. Option A verified.

---

## Problem P17: Bayes' theorem for rare disease screening

### 1. Given Information
- Sensitivity: $P(T^+ \mid D) = 0.99$
- Specificity: $P(T^- \mid D^c) = 0.95 \implies P(T^+ \mid D^c) = 1 - 0.95 = 0.05$
- Prevalence (Prior): $P(D) = 0.001 \implies P(D^c) = 0.999$
- Observation: Single positive test $T^+$
- Rounding: 4 decimal places

### 2. Method Choice and Justification
Bayes' Theorem for binary event: $P(D \mid T^+) = rac{P(T^+ \mid D) P(D)}{P(T^+ \mid D) P(D) + P(T^+ \mid D^c) P(D^c)}$.

### 3. Assumption Checks
Exhaustive binary partition $\{D, D^c\}$; known test performance characteristics.

### 4. Step-by-Step Computation & Code
```python
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
$$P(D \cap T^+) = 0.001 	imes 0.99 = 0.00099$$
$$P(D^c \cap T^+) = 0.999 	imes 0.05 = 0.04995$$
$$P(T^+) = 0.00099 + 0.04995 = 0.05094$$
$$P(D \mid T^+) = rac{0.00099}{0.05094} = rac{99}{5094} = rac{33}{1698} pprox 0.0194346$$
Rounding to 4 decimal places gives **0.0194**.

### 5. Conclusion in Context
Despite a 99% sensitive and 95% specific test, the posterior probability of disease given a positive test is only **0.0194** (1.94%), because false positives outnumber true positives by roughly 50 to 1.

### 6. Self-Audit & Dual Derivation
**Method 2 (Odds-Likelihood Formulation)**:
$$	ext{Prior Odds} = rac{0.001}{0.999} = rac{1}{999}$$
$$	ext{Bayes Factor} = rac{	ext{Sensitivity}}{1 - 	ext{Specificity}} = rac{0.99}{0.05} = 19.8$$
$$	ext{Posterior Odds} = rac{1}{999} 	imes 19.8 = rac{19.8}{999} = rac{198}{9990} = rac{11}{555} pprox 0.0198198$$
$$	ext{Posterior Probability} = rac{	ext{Odds}}{1 + 	ext{Odds}} = rac{11/555}{1 + 11/555} = rac{11}{566} pprox 0.0194346$$
Both formulations yield $rac{99}{5094} pprox 0.0194346$, rounding to **0.0194**.

---

## Problem P18: Sequential Bayesian updating with two independent tests

### 1. Given Information
- Prior: $P(D) = 0.001, P(D^c) = 0.999$
- Test 1 and Test 2 are conditionally independent given disease status.
- Each test has Sensitivity = 0.99 and Specificity = 0.95 ($FPR = 0.05$).
- Both tests return positive ($T_1^+, T_2^+$).
- Rounding: 4 decimal places

### 2. Method Choice and Justification
Bayes' Theorem with conditional independence: $P(D \mid T_1^+, T_2^+) = rac{P(D) P(T_1^+ \mid D) P(T_2^+ \mid D)}{P(D) P(T_1^+ \mid D) P(T_2^+ \mid D) + P(D^c) P(T_1^+ \mid D^c) P(T_2^+ \mid D^c)}$.

### 3. Assumption Checks
Conditional independence of test outcomes given infection state.

### 4. Step-by-Step Computation & Code
```python
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
$$P(D \cap T_1^+ \cap T_2^+) = 0.001 	imes (0.99)^2 = 0.001 	imes 0.9801 = 0.0009801$$
$$P(D^c \cap T_1^+ \cap T_2^+) = 0.999 	imes (0.05)^2 = 0.999 	imes 0.0025 = 0.0024975$$
$$P(T_1^+ \cap T_2^+) = 0.0009801 + 0.0024975 = 0.0034776$$
$$P(D \mid T_1^+, T_2^+) = rac{0.0009801}{0.0034776} = rac{9801}{34776} pprox 0.281832$$
Rounding to 4 decimal places gives **0.2818**.

### 5. Conclusion in Context
After two consecutive positive tests, the posterior probability of infection rises from 1.94% to **0.2818** (28.18%).

### 6. Self-Audit & Dual Derivation
**Method 2 (Sequential Updating with Intermediate Posterior as Prior)**:
Use the posterior of test 1 ($P_1 = rac{99}{5094} pprox 0.0194346$) as the prior for test 2:
$$P(D \mid T_1^+) = rac{99}{5094}, \quad P(D^c \mid T_1^+) = rac{4995}{5094}$$
Numerator: $rac{99}{5094} 	imes 0.99 = rac{98.01}{5094}$
Denominator: $rac{98.01}{5094} + rac{4995}{5094} 	imes 0.05 = rac{98.01 + 249.75}{5094} = rac{347.76}{5094}$
Posterior: $rac{98.01}{347.76} = rac{9801}{34776} pprox 0.2818323$.
Matches **0.2818** exactly.

---

## Problem P19: Multi-hypothesis Bayesian updating across three production lines

### 1. Given Information
- Line 1: $P(M_1) = 0.50, P(	ext{Defect} \mid M_1) = 0.02$
- Line 2: $P(M_2) = 0.30, P(	ext{Defect} \mid M_2) = 0.04$
- Line 3: $P(M_3) = 0.20, P(	ext{Defect} \mid M_3) = 0.10$
- Observation: Defective sensor
- Rounding: 4 decimal places

### 2. Method Choice and Justification
Law of Total Probability and Bayes' Rule for partitions: $P(M_3 \mid 	ext{Defect}) = rac{P(M_3) P(	ext{Defect} \mid M_3)}{\sum_{i=1}^3 P(M_i) P(	ext{Defect} \mid M_i)}$.

### 3. Assumption Checks
Mutually exclusive and exhaustive production lines $\sum P(M_i) = 1.0$.

### 4. Step-by-Step Computation & Code
```python
p_m = [0.50, 0.30, 0.20]
p_def_given_m = [0.02, 0.04, 0.10]

joints = [p * d for p, d in zip(p_m, p_def_given_m)]
# [0.010, 0.012, 0.020]

p_total_def = sum(joints) # 0.042
p_m3_given_def = joints[2] / p_total_def # 0.020 / 0.042 = 20 / 42 = 10 / 21 = 0.476190...
```
Calculation details:
$$P(M_1 \cap 	ext{Defect}) = 0.50 	imes 0.02 = 0.010$$
$$P(M_2 \cap 	ext{Defect}) = 0.30 	imes 0.04 = 0.012$$
$$P(M_3 \cap 	ext{Defect}) = 0.20 	imes 0.10 = 0.020$$
$$P(	ext{Defect}) = 0.010 + 0.012 + 0.020 = 0.042$$
$$P(M_3 \mid 	ext{Defect}) = rac{0.020}{0.042} = rac{20}{42} = rac{10}{21} pprox 0.476190$$
Rounding to 4 decimal places gives **0.4762**.

### 5. Conclusion in Context
The posterior probability that a defective sensor originated from Line $M_3$ is **0.4762** (47.62%).

### 6. Self-Audit & Dual Derivation
**Method 2 (Direct natural frequencies with base 1,000 sensors)**:
- Line 1: 500 sensors $	imes 2\% = 10$ defectives.
- Line 2: 300 sensors $	imes 4\% = 12$ defectives.
- Line 3: 200 sensors $	imes 10\% = 20$ defectives.
- Total defectives = $10 + 12 + 20 = 42$.
- Defectives from Line 3 = 20.
- Ratio = $rac{20}{42} = rac{10}{21} pprox 0.476190$.
Matches **0.4762** identically.

---

## Problem P20: Maximum A Posteriori (MAP) hypothesis selection

### 1. Given Information
- Prior probabilities: $P(H_1) = 0.60, P(H_2) = 0.30, P(H_3) = 0.10$
- Evidence likelihoods: $P(E \mid H_1) = 0.15, P(E \mid H_2) = 0.40, P(E \mid H_3) = 0.80$
- 4 multiple-choice options (A: $H_1$, B: $H_2$, C: $H_3$, D: tie)

### 2. Method Choice and Justification
Maximum A Posteriori (MAP) decision rule: $\hat{H}_{	ext{MAP}} = rg\max_{i} P(H_i \mid E) = rg\max_{i} [P(H_i) P(E \mid H_i)]$, because the marginal evidence $P(E)$ is an identical positive constant for all hypotheses.

### 3. Assumption Checks
Mutually exclusive and exhaustive hypotheses.

### 4. Step-by-Step Computation & Code
```python
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
- For $H_1$: $0.60 	imes 0.15 = 0.090$
- For $H_2$: $0.30 	imes 0.40 = 0.120$
- For $H_3$: $0.10 	imes 0.80 = 0.080$
Comparing joint probabilities:
$$0.120 > 0.090 > 0.080 \implies P(H_2 \mid E) > P(H_1 \mid E) > P(H_3 \mid E)$$
$H_2$ achieves the highest posterior probability ($41.38\%$), which corresponds to Option B.

### 5. Conclusion in Context
Hypothesis $H_2$ has the highest posterior probability (Option B).

### 6. Self-Audit & Dual Derivation
**Method 2 (Pairwise Odds Updates)**:
Prior odds $H_2$ vs $H_1$: $rac{0.30}{0.60} = 0.5$. Bayes factor: $rac{0.40}{0.15} = rac{8}{3} pprox 2.6667$.
Posterior odds $H_2$ vs $H_1$: $0.5 	imes rac{8}{3} = rac{4}{3} pprox 1.333 > 1 \implies H_2$ beats $H_1$.
Prior odds $H_2$ vs $H_3$: $rac{0.30}{0.10} = 3.0$. Bayes factor: $rac{0.40}{0.80} = 0.5$.
Posterior odds $H_2$ vs $H_3$: $3.0 	imes 0.5 = 1.5 > 1 \implies H_2$ beats $H_3$.
Since $H_2$ beats both $H_1$ and $H_3$, $H_2$ is confirmed as the MAP hypothesis (Option B).

---
