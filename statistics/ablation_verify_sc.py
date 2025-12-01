import matplotlib.pyplot as plt
import numpy as np

datasets = ["OpenBookQA", "Coin", "Letter", "AQuA"]
adv_cot_acc_arr = [84.2, 94.3, 75.4, 64.5]
adv_cot_verify_acc_arr = [87.4, 95.1, 75.6, 70.0]
adv_cot_sc_acc_arr = [86.0, 98.2, 78.4, 71.2]
adv_cot_verify_sc_acc_arr = [86.8, 99.2, 80.0, 71.6]

colors = ["#6C8EBF", "#82B366", "#B85450", "#9673A6"]

num_datasets = len(datasets)
bar_height = 0.2
index = np.arange(num_datasets)

fig, ax = plt.subplots(figsize=(8, 3.8))

ax.barh(index - 1.5 * bar_height, adv_cot_acc_arr, height=bar_height, color=colors[0], label="Optimized prompt")
ax.barh(index - 0.5 * bar_height, adv_cot_verify_acc_arr, height=bar_height, color=colors[1],
        label="Optimized prompt + Verification")
ax.barh(index + 0.5 * bar_height, adv_cot_sc_acc_arr, height=bar_height, color=colors[2],
        label="Optimized prompt + Self-consistency")
ax.barh(index + 1.5 * bar_height, adv_cot_verify_sc_acc_arr, height=bar_height, color=colors[3],
        label="Optimized prompt + Verify + Self-consistency")

ax.set_xlabel("Accuracy (%)", fontsize=12)
ax.set_yticks(index)
ax.set_yticklabels(datasets, fontsize=11)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    fontsize=9,
    frameon=False
)

plt.tight_layout()
plt.savefig("bar_horizontal_final.pdf", format="pdf", bbox_inches="tight")
plt.show()