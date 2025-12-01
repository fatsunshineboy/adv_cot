import matplotlib.pyplot as plt

MODEL_NAME = "gpt-4o-mini"

datasets_name = task_list = [
    "AQuA",
    "arc-c"
]

accuracy_temperature_0_0 = [80.3, 94.8]
accuracy_temperature_0_2 = [74.4, 94.6]
accuracy_temperature_0_4 = [78.3, 95.3]
accuracy_temperature_0_6 = [79.1, 95.3]
accuracy_temperature_0_8 = [73.6, 94.7]

temperatures = [0.0, 0.2, 0.4, 0.6, 0.8]

accuracy_aqua = [80.3, 74.4, 78.3, 79.1, 73.6]
accuracy_arcc = [94.8, 94.6, 95.3, 95.3, 94.7]

colors = ['#6C8EBF', '#82B366', '#B85450', '#FA6800']

plt.figure(figsize=(6, 4))
plt.plot(temperatures, accuracy_aqua, marker='o', linewidth=2, label="AQuA", color=colors[0])
plt.plot(temperatures, accuracy_arcc, marker='o', linewidth=2, label="ARC-C", color=colors[1])

plt.xlabel("Temperature (Proposer / Modifier)")
plt.ylabel("Accuracy (%)")
# plt.title("Temperature")
# plt.grid(True)
plt.legend()
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("temp_ablation.pdf")
plt.show()