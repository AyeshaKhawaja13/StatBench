import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

print("--- TOPIC A ---")
# A1: One-sample z-test for proportion
# H0: p = 0.05, n = 400, x = 28, p_hat = 28/400 = 0.07
p0 = 0.05
n = 400
x = 28
p_hat = x / n
se_a1 = np.sqrt(p0 * (1 - p0) / n)
z_a1 = (p_hat - p0) / se_a1
p_val_a1 = 1 - stats.norm.cdf(z_a1)
print(f"A1: p_hat={p_hat}, se={se_a1:.6f}, z={z_a1:.4f}, round(3)={z_a1:.3f}, p_val={p_val_a1:.4f}")

# A2: Paired t-test
# Patient diffs (After - Before)
d = np.array([-6, -4, -8, -5, -7, -3, -9, -4, -6, -8], dtype=float)
n_d = len(d)
mean_d = np.mean(d)
s_d = np.std(d, ddof=1)
t_a2 = mean_d / (s_d / np.sqrt(n_d))
p_val_a2 = 2 * stats.t.cdf(t_a2, df=n_d-1) # two-tailed
print(f"A2: mean_d={mean_d:.4f}, s_d={s_d:.4f}, t={t_a2:.4f}, round(3)={t_a2:.3f}, p_val={p_val_a2:.6f}")
# Also scipy.stats.ttest_1samp
t_check, p_check = stats.ttest_1samp(d, 0)
print(f"A2 check: t={t_check:.4f}, p={p_check:.6f}")

# A3: Welch's t-test
n1, m1, s1 = 12, 24.5, 1.8
n2, m2, s2 = 25, 21.0, 4.2
se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
t_a3 = (m1 - m2) / se_diff
# Welch-Satterthwaite df:
df_num = (s1**2/n1 + s2**2/n2)**2
df_den = (s1**2/n1)**2 / (n1 - 1) + (s2**2/n2)**2 / (n2 - 1)
df_welch = df_num / df_den
p_val_a3 = 2 * (1 - stats.t.cdf(abs(t_a3), df=df_welch))
print(f"A3: se_diff={se_diff:.6f}, t={t_a3:.4f}, round(3)={t_a3:.3f}, df={df_welch:.3f}, p_val={p_val_a3:.6f}")

# A4: Chi-square test
obs = np.array([[40, 60], [60, 40]])
chi2_a4, p_a4, dof_a4, expected = stats.chi2_contingency(obs, correction=False)
print(f"A4: chi2={chi2_a4:.4f}, round(3)={chi2_a4:.3f}, p={p_a4:.4f}, dof={dof_a4}")

print("\n--- TOPIC B ---")
# B1: 95% CI upper bound for mean (small n, unknown sigma)
n_b1 = 9
xbar_b1 = 50.0
s_b1 = 6.0
t_crit_b1 = stats.t.ppf(0.975, df=n_b1-1)
me_b1 = t_crit_b1 * (s_b1 / np.sqrt(n_b1))
ub_b1 = xbar_b1 + me_b1
print(f"B1: t_crit={t_crit_b1:.4f}, me={me_b1:.4f}, ub={ub_b1:.4f}, round(3)={ub_b1:.3f}")

# B2: 99% CI margin of error for diff of means (known sigmas)
sig1_sq, n1_b2 = 16.0, 15
sig2_sq, n2_b2 = 25.0, 20
z_crit_b2 = stats.norm.ppf(0.995)
se_b2 = np.sqrt(sig1_sq/n1_b2 + sig2_sq/n2_b2)
me_b2 = z_crit_b2 * se_b2
print(f"B2: z_crit={z_crit_b2:.4f}, se={se_b2:.4f}, me={me_b2:.4f}, round(3)={me_b2:.3f}")

# B3: 95% Wald CI lower bound for single proportion
n_b3 = 100
x_b3 = 35
p_hat_b3 = x_b3 / n_b3
z_crit_b3 = stats.norm.ppf(0.975)
se_b3 = np.sqrt(p_hat_b3 * (1 - p_hat_b3) / n_b3)
lb_b3 = p_hat_b3 - z_crit_b3 * se_b3
print(f"B3: z_crit={z_crit_b3:.4f}, se={se_b3:.4f}, lb={lb_b3:.6f}, round(4)={lb_b3:.4f}")

# B4: 90% Wald CI margin of error for diff of proportions
n1_b4, x1_b4 = 200, 80
n2_b4, x2_b4 = 250, 60
p1_b4 = x1_b4 / n1_b4
p2_b4 = x2_b4 / n2_b4
z_crit_b4 = stats.norm.ppf(0.95)
se_b4 = np.sqrt(p1_b4*(1-p1_b4)/n1_b4 + p2_b4*(1-p2_b4)/n2_b4)
me_b4 = z_crit_b4 * se_b4
print(f"B4: p1={p1_b4}, p2={p2_b4}, z_crit={z_crit_b4:.4f}, se={se_b4:.6f}, me={me_b4:.6f}, round(4)={me_b4:.4f}")

print("\n--- TOPIC C ---")
# C4: Multiple testing false positive probability
prob_no_error = (1 - 0.05)**20
prob_at_least_one = 1 - prob_no_error
print(f"C4: 1 - 0.95^20 = {prob_at_least_one:.6f}, round(3)={prob_at_least_one:.3f}")

print("\n--- TOPIC E ---")
# E1: Rare disease
p_d = 0.001
sens = 0.99
spec = 0.95
p_pos_d = sens
p_pos_dc = 1 - spec
p_pos = p_d * p_pos_d + (1 - p_d) * p_pos_dc
p_d_given_pos = (p_d * p_pos_d) / p_pos
print(f"E1: P(D|pos) = {p_d_given_pos:.6f}, round(4)={p_d_given_pos:.4f}")

# E2: Two sequential tests
p_pos2_d = sens**2
p_pos2_dc = (1 - spec)**2
p_pos2 = p_d * p_pos2_d + (1 - p_d) * p_pos2_dc
p_d_given_2pos = (p_d * p_pos2_d) / p_pos2
print(f"E2: P(D|2pos) = {p_d_given_2pos:.6f}, round(4)={p_d_given_2pos:.4f}")

# E3: Three machines
# P(M1)=0.5, P(D|M1)=0.02
# P(M2)=0.3, P(D|M2)=0.04
# P(M3)=0.2, P(D|M3)=0.10
p_m = [0.5, 0.3, 0.2]
p_def_given_m = [0.02, 0.04, 0.10]
joints = [p * d for p, d in zip(p_m, p_def_given_m)]
total_def = sum(joints)
p_m3_given_def = joints[2] / total_def
print(f"E3: P(M3|def) = {joints[2]}/{total_def} = {p_m3_given_def:.6f}, round(4)={p_m3_given_def:.4f}")

# E4: MAP hypothesis
p_prior = [0.60, 0.30, 0.10]
p_lik = [0.15, 0.40, 0.80]
unnorm = [pr * l for pr, l in zip(p_prior, p_lik)]
total_e = sum(unnorm)
posteriors = [u / total_e for u in unnorm]
print(f"E4: unnormalized={unnorm}, posteriors={posteriors}, MAP index={np.argmax(posteriors)}")
