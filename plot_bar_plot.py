import numpy as np
import matplotlib.pyplot as plt

# Define experiments and result files
experiments = ["acc_only", "fsr_only", "kinesthetic_only", "combined"]
result_files = [f"results/{name}_results.npz" for name in experiments]

# Store means and stds for each classifier
logreg_means, logreg_stds = [], []
svm_means, svm_stds = [], []

# Load each result file and extract accuracy
for file in result_files:
    data = np.load(file)
    logreg_acc = data['logreg_accuracy']
    svm_acc = data['svm_accuracy']

    logreg_means.append(np.mean(logreg_acc))
    logreg_stds.append(np.std(logreg_acc))

    svm_means.append(np.mean(svm_acc))
    svm_stds.append(np.std(svm_acc))

# === Plotting ===
x = np.arange(len(experiments))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, logreg_means, width, yerr=logreg_stds, label='Logistic Regression', capsize=4)
ax.bar(x + width / 2, svm_means, width, yerr=svm_stds, label='SVM', capsize=4)

ax.set_ylabel('Accuracy')
ax.set_title('Classifier Accuracy per Feature Group')
ax.set_xticks(x)
ax.set_xticklabels(experiments, rotation=20)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("results/classifier_accuracy_comparison.png")
plt.show()
