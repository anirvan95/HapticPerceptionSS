"""
Time-series classifier for haptic perception data
Classifies letters (C, D, Q) based on tactile and kinesthetic sensor data
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.fft import fft
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class HapticTimeSeriesClassifier:
    """
    Time-series classifier for haptic perception data
    """

    def __init__(self, data_path="dataset/processed_data.npz"):
        self.data_path = data_path
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.X_test = None
        self.y_test = None

    def parse_filename(self, filename):
        """
        Parse filename to extract metadata
        Format: {letter}_{stiffness}_{shape}_{frequency}_{iteration}.npy
        """
        parts = filename.replace(".npy", "").split("_")
        if len(parts) >= 5:
            return {
                "letter": parts[0],
                "stiffness": parts[1],
                "shape": parts[2],
                "frequency": float(parts[3]),
                "iteration": int(parts[4]),
            }
        return None

    def extract_features_from_arrays(self, fsr0, fsr1, acc, kinesthetic):
        """
        Extract comprehensive features from all sensor modalities
        """
        print("Extracting features from sensor data...")

        n_samples = fsr0.shape[0]
        all_features = []

        for i in range(n_samples):
            sample_features = []

            # FSR0 features (16x16 sensor array over time)
            fsr0_sample = fsr0[i]  # (1111, 16, 16)
            sample_features.extend(self.extract_fsr_features(fsr0_sample, "FSR0"))

            # FSR1 features (16x16 sensor array over time)
            fsr1_sample = fsr1[i]  # (1111, 16, 16)
            sample_features.extend(self.extract_fsr_features(fsr1_sample, "FSR1"))

            # Accelerometer features (16x3 over time)
            acc_sample = acc[i]  # (1111, 16, 3)
            sample_features.extend(self.extract_acc_features(acc_sample))

            # Kinesthetic features (3D pose over time)
            kin_sample = kinesthetic[i]  # (50, 3)
            sample_features.extend(self.extract_kinesthetic_features_new(kin_sample))

            all_features.append(sample_features)

        features_array = np.array(all_features)
        print(f"Extracted feature matrix shape: {features_array.shape}")

        return features_array

    def extract_fsr_features(self, fsr_data, prefix):
        """
        Extract features from FSR sensor array data (1111, 16, 16)
        """
        features = []

        # Flatten spatial dimensions to get (1111, 256) - time x sensors
        fsr_flat = fsr_data.reshape(fsr_data.shape[0], -1)

        # Statistical features across time
        features.extend(
            [
                np.mean(fsr_flat, axis=0),  # Mean per sensor
                np.std(fsr_flat, axis=0),  # Std per sensor
                np.max(fsr_flat, axis=0),  # Max per sensor
                np.min(fsr_flat, axis=0),  # Min per sensor
            ]
        )

        # Global statistical features
        features.extend(
            [
                np.mean(fsr_data),  # Global mean
                np.std(fsr_data),  # Global std
                np.max(fsr_data),  # Global max
                np.min(fsr_data),  # Global min
            ]
        )

        # Spatial patterns (mean over time for each position)
        spatial_mean = np.mean(fsr_data, axis=0)  # (16, 16)
        features.extend(
            [
                spatial_mean.flatten(),  # Spatial distribution
                np.sum(spatial_mean, axis=0),  # Column sums
                np.sum(spatial_mean, axis=1),  # Row sums
            ]
        )

        # Temporal dynamics (global signal over time)
        temporal_signal = np.mean(fsr_flat, axis=1)  # (1111,)
        if len(temporal_signal) > 4:
            # FFT features
            fft_vals = np.abs(fft(temporal_signal))
            features.extend(
                [
                    np.mean(fft_vals[:50]),  # Low frequency content
                    np.std(fft_vals[:50]),
                    np.max(fft_vals[:50]),
                ]
            )

        return np.concatenate(
            [f.flatten() if hasattr(f, "flatten") else [f] for f in features]
        )

    def extract_acc_features(self, acc_data):
        """
        Extract features from accelerometer data (1111, 16, 3)
        """
        features = []

        # Reshape to (1111, 48) - time x (16 sensors * 3 axes)
        acc_flat = acc_data.reshape(acc_data.shape[0], -1)

        # Statistical features across time
        features.extend(
            [
                np.mean(acc_flat, axis=0),  # Mean per sensor-axis
                np.std(acc_flat, axis=0),  # Std per sensor-axis
                np.max(acc_flat, axis=0),  # Max per sensor-axis
                np.min(acc_flat, axis=0),  # Min per sensor-axis
            ]
        )

        # Global features
        features.extend(
            [
                np.mean(acc_data),
                np.std(acc_data),
                np.max(acc_data),
                np.min(acc_data),
            ]
        )

        # Per-axis features
        for axis in range(3):
            axis_data = acc_data[:, :, axis]  # (1111, 16)
            features.extend(
                [
                    np.mean(axis_data),
                    np.std(axis_data),
                    np.max(axis_data),
                    np.min(axis_data),
                ]
            )

        # Magnitude features
        magnitude = np.sqrt(np.sum(acc_data**2, axis=2))  # (1111, 16)
        features.extend(
            [
                np.mean(magnitude),
                np.std(magnitude),
                np.max(magnitude),
                np.min(magnitude),
            ]
        )

        return np.concatenate(
            [f.flatten() if hasattr(f, "flatten") else [f] for f in features]
        )

    def extract_kinesthetic_features_new(self, kin_data):
        """
        Extract features from kinesthetic data (50, 3)
        """
        features = []

        # Statistical features for each axis
        for axis in range(3):
            axis_data = kin_data[:, axis]
            features.extend(
                [
                    np.mean(axis_data),
                    np.std(axis_data),
                    np.max(axis_data),
                    np.min(axis_data),
                    np.median(axis_data),
                ]
            )

        # Global features
        features.extend(
            [
                np.mean(kin_data),
                np.std(kin_data),
                np.max(kin_data),
                np.min(kin_data),
            ]
        )

        # Motion features
        if kin_data.shape[0] > 1:
            # Velocity (differences)
            velocity = np.diff(kin_data, axis=0)
            features.extend(
                [
                    np.mean(velocity),
                    np.std(velocity),
                    np.max(velocity),
                    np.min(velocity),
                ]
            )

            # Path length
            path_length = np.sum(np.sqrt(np.sum(velocity**2, axis=1)))
            features.append(path_length)

        return np.array(features)

    def extract_tactile_features(self, tactile_data):
        """
        Extract features from tactile sensor data (accelerometer + FSR)
        """
        features = []

        if not tactile_data:
            return np.array([])

        # Extract time series data
        timestamps = []
        acc_data = []
        fsr1_data = []
        fsr2_data = []

        for timestamp, acc, fsr1, fsr2 in tactile_data:
            timestamps.append(timestamp)
            acc_data.append(acc.flatten())  # 16x3 -> 48 features
            fsr1_data.append(fsr1.flatten())  # 16x16 -> 256 features
            fsr2_data.append(fsr2.flatten())  # 16x16 -> 256 features

        timestamps = np.array(timestamps)
        acc_data = np.array(acc_data)
        fsr1_data = np.array(fsr1_data)
        fsr2_data = np.array(fsr2_data)

        # Statistical features for accelerometer data
        if acc_data.size > 0:
            features.extend(
                [
                    np.mean(acc_data, axis=0),  # Mean across time
                    np.std(acc_data, axis=0),  # Standard deviation
                    np.max(acc_data, axis=0),  # Maximum values
                    np.min(acc_data, axis=0),  # Minimum values
                    np.median(acc_data, axis=0),  # Median values
                ]
            )

        # Statistical features for FSR data (combined)
        fsr_combined = np.concatenate([fsr1_data, fsr2_data], axis=1)
        if fsr_combined.size > 0:
            features.extend(
                [
                    np.mean(fsr_combined, axis=0),  # Mean across time
                    np.std(fsr_combined, axis=0),  # Standard deviation
                    np.max(fsr_combined, axis=0),  # Maximum values
                    np.min(fsr_combined, axis=0),  # Minimum values
                ]
            )

        # Temporal features
        if len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            sampling_rate = len(timestamps) / duration if duration > 0 else 0
            features.extend([duration, sampling_rate, len(timestamps)])

        # Frequency domain features (FFT) for key sensors
        if acc_data.size > 0 and len(acc_data) > 4:
            # FFT for first few accelerometer channels
            for i in range(min(6, acc_data.shape[1])):
                fft_values = np.abs(fft(acc_data[:, i]))
                features.extend(
                    [np.mean(fft_values), np.std(fft_values), np.max(fft_values)]
                )

        return np.concatenate(
            [f.flatten() if hasattr(f, "flatten") else [f] for f in features]
        )

    def extract_kinesthetic_features(self, kinesthetic_data):
        """
        Extract features from kinesthetic data (6DOF pose)
        """
        features = []

        if not kinesthetic_data or len(kinesthetic_data) == 0:
            return np.array([])

        # Extract pose data (assuming similar structure to tactile)
        poses = []
        timestamps = []

        for item in kinesthetic_data:
            if isinstance(item, tuple) and len(item) >= 2:
                timestamps.append(item[0])
                if isinstance(item[1], np.ndarray):
                    poses.append(item[1].flatten())

        if poses:
            poses = np.array(poses)
            timestamps = np.array(timestamps)

            # Statistical features
            features.extend(
                [
                    np.mean(poses, axis=0),
                    np.std(poses, axis=0),
                    np.max(poses, axis=0),
                    np.min(poses, axis=0),
                ]
            )

            # Temporal features
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                features.extend([duration, len(timestamps)])

        return np.concatenate(
            [f.flatten() if hasattr(f, "flatten") else [f] for f in features]
        )

    def load_and_process_data(self):
        """
        Load the processed .npz data and extract features
        """
        print("Loading processed haptic data...")

        if not os.path.exists(self.data_path):
            print(f"Data file not found: {self.data_path}")
            return False

        # Load data
        data = np.load(self.data_path)

        # Extract arrays
        fsr0 = data["fsr0"]  # (450, 1111, 16, 16)
        fsr1 = data["fsr1"]  # (450, 1111, 16, 16)
        acc = data["acc"]  # (450, 1111, 16, 3)
        kinesthetic = data["kinesthetic"]  # (450, 50, 3)
        labels = data[
            "labels"
        ]  # (450, 5) - [letter, stiffness, shape, frequency, iteration]

        print("Data shapes:")
        print(f"  FSR0: {fsr0.shape}")
        print(f"  FSR1: {fsr1.shape}")
        print(f"  Accelerometer: {acc.shape}")
        print(f"  Kinesthetic: {kinesthetic.shape}")
        print(f"  Labels: {labels.shape}")

        # Extract letter labels (first column)
        self.y = labels[:, 0]  # Letter labels: C, D, Q

        print(
            f"Label distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

        # Extract features
        self.X = self.extract_features_from_arrays(fsr0, fsr1, acc, kinesthetic)

        # Encode labels
        self.y = self.label_encoder.fit_transform(self.y)

        return True

    def train_models(self, test_size=0.2, random_state=42):
        """
        Train multiple classification models
        """
        if self.X is None or self.y is None:
            print("No data available for training")
            return False

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        model_configs = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
            "SVM": SVC(kernel="rbf", random_state=random_state),
            "Logistic Regression": LogisticRegression(
                random_state=random_state, max_iter=1000
            ),
        }

        print("Training models...")
        results = {}

        for name, model in model_configs.items():
            print(f"\nTraining {name}...")

            # Create pipeline
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])

            # Train model
            pipeline.fit(X_train, y_train)

            # Evaluate
            train_score = pipeline.score(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)

            # Cross-validation
            cv_scores = cross_val_score(
                pipeline,
                self.X,
                self.y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
                scoring="accuracy",
            )

            results[name] = {
                "model": pipeline,
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "cv_scores": cv_scores,
            }

            print(f"Train Accuracy: {train_score:.3f}")
            print(f"Test Accuracy: {test_score:.3f}")
            print(f"CV Mean ± Std: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Store results
        self.models = results

        return True

    def evaluate_models(self):
        """
        Detailed evaluation of trained models
        """
        if not self.models:
            print("No trained models available")
            return

        print("\n" + "=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)

        # Summary table
        summary_data = []
        for name, results in self.models.items():
            summary_data.append(
                {
                    "Model": name,
                    "Train Acc": f"{results['train_accuracy']:.3f}",
                    "Test Acc": f"{results['test_accuracy']:.3f}",
                    "CV Mean": f"{results['cv_mean']:.3f}",
                    "CV Std": f"{results['cv_std']:.3f}",
                }
            )

        summary_df = pl.DataFrame(summary_data)
        print(summary_df)

        # Best model
        best_model_name = max(
            self.models.keys(), key=lambda x: self.models[x]["cv_mean"]
        )
        best_model = self.models[best_model_name]["model"]

        print(f"\nBest Model: {best_model_name}")
        print(
            f"Cross-validation Score: {self.models[best_model_name]['cv_mean']:.3f} ± {self.models[best_model_name]['cv_std']:.3f}"
        )

        # Detailed classification report for best model
        y_pred = best_model.predict(self.X_test)

        print(f"\nDetailed Classification Report for {best_model_name}:")
        print("-" * 50)
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix for {best_model_name}:")
        print("-" * 30)

        # Create confusion matrix display using polars
        cm_data = {}
        for i, col_name in enumerate(target_names):
            cm_data[col_name] = cm[:, i]
        cm_df = pl.DataFrame(cm_data)
        cm_df = cm_df.with_columns(pl.Series("Index", target_names))
        cm_df = cm_df.select(["Index"] + list(target_names))
        print(cm_df)

        return best_model_name, best_model

    def visualize_results(self):
        """
        Create visualizations matching the original plot structure
        """
        if not self.models:
            return

        # Set style for better plots
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with same 2x2 layout as original
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Accuracy comparison
        model_names = list(self.models.keys())
        train_accs = [self.models[name]["train_accuracy"] for name in model_names]
        test_accs = [self.models[name]["test_accuracy"] for name in model_names]
        cv_means = [self.models[name]["cv_mean"] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        bars1 = ax1.bar(
            x - width, train_accs, width, label="Train", alpha=0.8, color="#3498db"
        )
        bars2 = ax1.bar(x, test_accs, width, label="Test", alpha=0.8, color="#e67e22")
        bars3 = ax1.bar(
            x + width, cv_means, width, label="CV Mean", alpha=0.8, color="#2ecc71"
        )
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Model Performance Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha="right")
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Cross-validation scores distribution
        cv_data = [self.models[name]["cv_scores"] for name in model_names]
        bp = ax2.boxplot(cv_data, labels=model_names, patch_artist=True)

        # Color the boxplots
        colors = ["#3498db", "#e67e22", "#2ecc71", "#f39c12"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title("Cross-Validation Score Distribution")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

        # 3. Label distribution
        label_counts = dict(
            zip(
                *np.unique(
                    self.label_encoder.inverse_transform(self.y),
                    return_counts=True,
                )
            )
        )
        bars = ax3.bar(
            label_counts.keys(),
            label_counts.values(),
            color=["#3498db", "#e67e22", "#2ecc71"],
            alpha=0.8,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )
        ax3.set_title("Class Distribution in Dataset")
        ax3.set_xlabel("Letters")
        ax3.set_ylabel("Count")
        ax3.grid(True, alpha=0.3)

        # 4. Confusion matrix for best model
        best_model_name = max(
            self.models.keys(), key=lambda x: self.models[x]["cv_mean"]
        )
        best_model = self.models[best_model_name]["model"]
        y_pred = best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        # Create heatmap
        im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
        ax4.figure.colorbar(im, ax=ax4)
        ax4.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            title=f"Confusion Matrix - {best_model_name}",
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax4.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig("haptic_classification_results.png", dpi=300, bbox_inches="tight")
        plt.show()

    def run_full_pipeline(self):
        """
        Run the complete classification pipeline
        """
        print("Starting Haptic Time-Series Classification Pipeline")
        print("=" * 60)

        # Load and process data
        if not self.load_and_process_data():
            print("Failed to load data")
            return False

        # Train models
        if not self.train_models():
            print("Failed to train models")
            return False

        # Evaluate models
        best_model_name, best_model = self.evaluate_models()

        # Visualize results
        self.visualize_results()

        print("\nPipeline completed successfully!")
        print(f"Best performing model: {best_model_name}")

        return True, best_model_name, best_model


# Usage example
if __name__ == "__main__":
    classifier = HapticTimeSeriesClassifier()
    success, best_model_name, best_model = classifier.run_full_pipeline()

    if success:
        print("\nClassification pipeline completed successfully!")
        print(f"Best model: {best_model_name}")
    else:
        print("Classification pipeline failed!")
