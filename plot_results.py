import csv
import matplotlib.pyplot as plt
import os


def load_csv(path):
    labeled, acc = [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labeled.append(int(row["labeled_samples"]))
            acc.append(float(row["accuracy"]))
    return labeled, acc


# Load results
rand_labels, rand_acc = load_csv("results/random.csv")
ent_labels, ent_acc = load_csv("results/entropy.csv")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(rand_labels, rand_acc, marker="o", label="Random Sampling")
plt.plot(ent_labels, ent_acc, marker="o", label="Entropy Sampling")
plt.xlabel("Number of Labeled Samples")
plt.ylabel("Test Accuracy")
plt.title("Active Learning vs Random Sampling")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save
os.makedirs("results", exist_ok=True)
plt.savefig("results/active_learning_comparison.png", dpi=300)
plt.show()
