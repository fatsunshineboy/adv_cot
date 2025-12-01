import matplotlib.pyplot as plt
import numpy as np

# datasets
datasets = ["ARC-c", "Sports", "BoolQ", "Letter", "AQuA"]
x = np.arange(len(datasets))

adv_cot_acc = [95.3, 82.3, 81.0, 90.5, 79.5]
adv_cot_wf_acc = [95.8, 78.6, 80.0, 84.8, 76.8]
adv_cot_wv_acc = [94.1, 74.0, 79.5, 84.3, 76.1]

adv_cot_loss = [13.3, 10.3, 10.3, 10.0, 11.6]
adv_cot_wf_loss = [15.0, 11.9, 14.6, 16.0, 17.33]

colors = ['#6C8EBF', '#82B366', '#B85450']
hatches = ['', '//', 'xx']

bar_width = 0.18
offset = bar_width + 0.02

plt.rc('font', family='Lucida Console', size=10)

fig, ax = plt.subplots(figsize=(6, 3))

ax.bar(x - offset, adv_cot_acc, width=bar_width, color=colors[0], edgecolor='black', hatch=hatches[0], label='Adv-CoT')
ax.bar(x, adv_cot_wf_acc, width=bar_width, color=colors[1], edgecolor='black', hatch=hatches[1], label='w/o Feedback')
ax.bar(x + offset, adv_cot_wv_acc, width=bar_width, color=colors[2], edgecolor='black', hatch=hatches[2],
       label='w/o Verify')

ax.set_ylabel('Accuracy', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylim(70, 100)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#000000')
ax.tick_params(axis='x', colors='#000000')
ax.tick_params(axis='y', colors='#000000')

ax.legend(frameon=False, fontsize=9, loc='upper right')
plt.tight_layout()
fig.savefig("./ablation_accuracy.pdf", format='pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6, 3))

ax.bar(x - bar_width / 2 - 0.01, adv_cot_loss, width=bar_width, color=colors[0], edgecolor='black', hatch=hatches[0],
       label='Adv-CoT')
ax.bar(x + bar_width / 2 + 0.01, adv_cot_wf_loss, width=bar_width, color=colors[1], edgecolor='black', hatch=hatches[1],
       label='w/o Feedback')

ax.set_ylabel('Training Steps', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylim(8, 20)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#000000')
ax.tick_params(axis='x', colors='#000000')
ax.tick_params(axis='y', colors='#000000')

ax.legend(frameon=False, fontsize=9, loc='upper right')
plt.tight_layout()
fig.savefig("./ablation_training.pdf", format='pdf', bbox_inches='tight')
plt.show()