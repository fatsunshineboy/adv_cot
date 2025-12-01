import numpy as np
from scipy import stats

# =========================
# 1. datasets
# =========================

cot_gpt35 = [78.2, 73.8, 85.6, 88.4, 87.3, 69.2, 65.6, 77.2, 82.2, 83.3, 62.2, 97.6]
advcot_gpt35 = [78.2, 73.3, 86.8, 89.5, 91.8, 73.5, 76.0, 98.4, 85.9, 85.3, 66.1, 99.1]

cot_gpt4o = [84.9, 77.4, 94.1, 94.3, 92.3, 77.1, 86.8, 100, 93.4, 93.6, 77.1, 98.1]
advcot_gpt4o = [84.8, 79.3, 94.8, 95.3, 93.1, 77.5, 91.0, 100, 93.7, 94.1, 79.9, 98.6]

cot_llama = [73.6, 61.3, 77.0, 80.3, 86.8, 59.4, 57.1, 89.6, 79.6, 86.7, 48.4, 97.5]
advcot_llama = [74.0, 62.3, 74.6, 81.4, 83.8, 62.5, 61.6, 89.0, 82.3, 86.9, 53.1, 97.6]

models = {
    "GPT-3.5": (cot_gpt35, advcot_gpt35),
    "GPT-4o-mini": (cot_gpt4o, advcot_gpt4o),
    "Llama-3-8B": (cot_llama, advcot_llama)
}


# =========================
# 2. paired t-tests, 10,000-sample bootstrap procedures, and Cohen’s d for dependent samples
# =========================

def cohen_d_paired(cot, adv):
    """
    Paired samples Cohen's d
    d = mean(diff) / std(diff)
    """
    diff = np.array(adv) - np.array(cot)
    return diff.mean() / diff.std(ddof=1)


def significance_test(cot, advcot):
    diff = np.array(advcot) - np.array(cot)

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(advcot, cot)

    # Bootstrap 95% CI
    boot_means = []
    for _ in range(10000):
        sample = np.random.choice(diff, size=len(diff), replace=True)
        boot_means.append(sample.mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # Cohen's d (paired version)
    d_value = cohen_d_paired(cot, advcot)

    return {
        "mean_diff": diff.mean(),
        "t_test_p": p_ttest,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohen_d": d_value
    }


# =========================
# 3. output
# =========================

for name, (cot, advcot) in models.items():
    res = significance_test(cot, advcot)
    print(f"\n=== {name} ===")
    print("Mean improvement:", round(res["mean_diff"], 3))
    print("Paired t-test p-value:", res["t_test_p"])
    print("Bootstrap 95% CI:", (round(res["ci_low"], 3), round(res["ci_high"], 3)))
    print("Cohen's d:", round(res["cohen_d"], 3))