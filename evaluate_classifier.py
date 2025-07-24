from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif

def evaluate_classifiers(
        X, y,
        save_filename=None,
        plot_pca=False,
        class_labels=None,
        cm_prefix="confmat",
        experiment_name="experiment"
):
    """
    Evaluate Logistic Regression, SVM, and Random Forest with Stratified 5-Fold CV.
    Adds PCA plot and saves global confusion matrices (SVG).

    Args:
        X (ndarray): Feature matrix
        y (ndarray): Labels
        save_filename (str): File to save accuracy results
        plot_pca (bool): Whether to show PCA 2D plot
        class_labels (list or None): Labels for confusion matrix
        cm_prefix (str): Prefix for confusion matrix filenames
        experiment_name (str): Name of the experiment (e.g., 'fsr only')
    """


    print(f"\n=== Starting Experiment: {experiment_name} ===")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if plot_pca:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=30)
        plt.title(f"{experiment_name} - PCA Projection")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{experiment_name.replace(' ', '_')}_pca.svg")
        plt.show()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_logreg, acc_svm, acc_rf = [], [], []

    global_cm_logreg = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)
    global_cm_svm = np.zeros_like(global_cm_logreg)
    global_cm_rf = np.zeros_like(global_cm_logreg)

    fold = 1
    for train_idx, test_idx in kf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Logistic Regression
        logreg = LogisticRegression(max_iter=1000, random_state=0)
        logreg.fit(X_train, y_train)
        y_pred_lr = logreg.predict(X_test)
        acc_logreg.append(accuracy_score(y_test, y_pred_lr))
        global_cm_logreg += confusion_matrix(y_test, y_pred_lr, labels=np.unique(y))

        # SVM
        svm = SVC(kernel='rbf', gamma='scale', random_state=0)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        acc_svm.append(accuracy_score(y_test, y_pred_svm))
        global_cm_svm += confusion_matrix(y_test, y_pred_svm, labels=np.unique(y))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf.append(accuracy_score(y_test, y_pred_rf))
        global_cm_rf += confusion_matrix(y_test, y_pred_rf, labels=np.unique(y))

        print(
            f"{experiment_name} - Fold {fold}: LR = {acc_logreg[-1]:.4f}, SVM = {acc_svm[-1]:.4f}, RF = {acc_rf[-1]:.4f}")
        fold += 1

    # === Final Report ===
    print(f"\n=== Final Results for {experiment_name} ===")
    print(f"Logistic Regression: {np.mean(acc_logreg):.4f} ± {np.std(acc_logreg):.4f}")
    print(f"SVM: {np.mean(acc_svm):.4f} ± {np.std(acc_svm):.4f}")
    print(f"Random Forest: {np.mean(acc_rf):.4f} ± {np.std(acc_rf):.4f}")

    labels_display = class_labels if class_labels is not None else np.unique(y)
    safe_exp_name = experiment_name.replace(' ', '_')

    for clf_name, cm, cmap, color, acc in zip(
            ["Logistic Regression", "SVM", "Random Forest"],
            [global_cm_logreg, global_cm_svm, global_cm_rf],
            ['Blues', 'Oranges', 'Greens'],
            ['lr', 'svm', 'rf'],
            [acc_logreg, acc_svm, acc_rf]
    ):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_display)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, xticks_rotation=45, cmap=cmap, colorbar=False)
        plt.title(f"{experiment_name} - Confusion Matrix ({clf_name})")
        plt.tight_layout()
        svg_file = f"results/{safe_exp_name}_{cm_prefix}_{color}.png"
        plt.savefig(svg_file)
        print(f"Saved {clf_name} confusion matrix to: {svg_file}")
        plt.close(fig)

    if save_filename:
        save_path = f"results/{safe_exp_name}_{save_filename}"
        with open(save_path, 'w') as f:
            f.write(f"Classification Report: {experiment_name}\n")
            f.write(f"Logistic Regression: {np.mean(acc_logreg):.4f} ± {np.std(acc_logreg):.4f}\n")
            f.write(f"SVM: {np.mean(acc_svm):.4f} ± {np.std(acc_svm):.4f}\n")
            f.write(f"Random Forest: {np.mean(acc_rf):.4f} ± {np.std(acc_rf):.4f}\n\n")

            f.write("=== Confusion Matrices ===\n\n")
            f.write("Logistic Regression:\n")
            np.savetxt(f, global_cm_logreg, fmt='%d')
            f.write("\nSVM:\n")
            np.savetxt(f, global_cm_svm, fmt='%d')
            f.write("\nRandom Forest:\n")
            np.savetxt(f, global_cm_rf, fmt='%d')
        print(f"Accuracy summary saved to: {save_path}")

    return {
        'logreg_accuracy': acc_logreg,
        'svm_accuracy': acc_svm,
        'rf_accuracy': acc_rf,
        'conf_matrix_logreg': global_cm_logreg,
        'conf_matrix_svm': global_cm_svm,
        'conf_matrix_rf': global_cm_rf
    }


# ############################ Experiment ##################################
data = np.load('features_all_modalities.npz')
object_labels = data['object_labels']

acc_features = np.concatenate([
    data['acc_mean'][:, None],
    data['acc_std'][:, None],
    data['acc_min'][:, None],
    data['acc_max'][:, None],
    data['acc_range'][:, None],
    data['acc_skew'][:, None],
    data['acc_kurt'][:, None],
    data['mean_over_sensors'][:, None],   # shape: (N, 3)
    data['std_over_sensors'][:, None],    # shape: (N, 3)
    data['acc_energy'][:, None],
    data['acc_entropy'][:, None]
], axis=1)

fsr_features = np.concatenate([
    data['gstat_fsr0'],
    data['gstat_fsr1'],
    data['qstat_fsr0'],
    data['qstat_fsr1'],
    data['grad_mag'][:, None],
    data['sav0_mean'][:, None],
    data['sav0_slope'][:, None],
    data['sav1_mean'][:, None],
    data['sav1_slope'][:, None],
    data['fft_bands_diff'],
    data['fft_centroid_diff'][:, None],       # (N, 3)
    data['fft_energy_diff'][:, None],
    data['fft_flatness_diff'][:, None],
    data['fft_entropy_diff'][:, None]
], axis=1)


kin_features = np.concatenate([
    data['kin_var'],         # (N, 3)
    data['kin_skew'],        # (N, 3)
    data['kin_kurt'],        # (N, 3)
    data['features_fsr_kin'] # (N, 12)
], axis=1)  # Resulting shape: (N, 21)

# -------- Feature Selection per Modality --------
def select_top_k_features(X, y, k):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X_scaled, y)
    return X_new

acc_features_sel = select_top_k_features(acc_features, object_labels, k=5)
fsr_features_sel = select_top_k_features(fsr_features, object_labels, k=10)
kin_features_sel = select_top_k_features(kin_features, object_labels, k=5)

# -------- Combine selected features and evaluate --------
all_combined = np.concatenate([acc_features_sel, fsr_features_sel, kin_features_sel], axis=1)

acc_results = evaluate_classifiers(acc_features, object_labels, save_filename='class_stats', experiment_name='acc_only')
fsr_results = evaluate_classifiers(fsr_features, object_labels, save_filename='class_stats', experiment_name='fsr_only')
kin_results = evaluate_classifiers(kin_features, object_labels, save_filename='class_stats', experiment_name='kinesthetic_only')
comb_results = evaluate_classifiers(all_combined, object_labels, save_filename='class_stats', experiment_name='combined_top_features')

# Save each result dictionary with the correct name
np.savez('results/acc_only_results.npz', **acc_results)
np.savez('results/fsr_only_results.npz', **fsr_results)
np.savez('results/kinesthetic_only_results.npz', **kin_results)
np.savez('results/combined_results.npz', **comb_results)