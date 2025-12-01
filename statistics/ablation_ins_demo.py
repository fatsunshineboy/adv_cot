import matplotlib.pyplot as plt
import numpy as np

datasets = ["AQuA", "Letter", "ARC-c"]

cot_accuracy = [77.1, 86.8, 94.3]
acc_instr = [77.9, 90.4, 94.3]
acc_demo = [79.1, 89.2, 94.6]
acc_both = [79.9, 91.0, 95.3]

colors = ['#6C8EBF', '#82B366', '#B85450', '#FA6800']

x = np.arange(len(datasets))
width = 0.2

plt.figure(figsize=(8, 6))

plt.bar(x - 1.5 * width, cot_accuracy, width, label="CoT baseline", color=colors[0])
plt.bar(x - 0.5 * width, acc_instr, width, label="Instruction only", color=colors[1])
plt.bar(x + 0.5 * width, acc_demo, width, label="Demonstrations only", color=colors[2])
plt.bar(x + 1.5 * width, acc_both, width, label="Joint optimization", color=colors[3])

plt.xticks(x, datasets)
plt.ylabel("Accuracy (%)")
# plt.title("Ablation on Instruction vs. Demonstration Optimization")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("instr_demo_bar.pdf")
plt.show()