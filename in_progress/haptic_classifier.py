"""
Unified haptic perception classifier combining best approaches
Multi-output classification with comprehensive feature extraction and dimensionality reduction
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.fft import fft
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

warnings.filterwarnings("ignore")


class HapticClassifier:
    """
    Unified haptic perception classifier
    - Multi-output approach (separate models per factor)
    - Comprehensive feature extraction with PCA reduction
    - Ensemble methods for robustness
    """

    def __init__(
        self, data_path="dataset/processed_data.npz", feature_mode="comprehensive"
    ):
        self.data_path = data_path
        self.feature_mode = feature_mode  # "simple" or "comprehensive"
        self.X = None
        self.X_reduced = None
        self.label_encoders = {}
        self.models = {}
        self.ensemble_models = {}
        self.pca = None
        self.feature_selector = None
        self.scaler = StandardScaler()

    def extract_comprehensive_features(self, fsr0, fsr1, acc, kinesthetic):
        """
        Extract comprehensive feature set combining spatial, temporal, and spectral features
        All features are properly normalized to ensure consistent scales
        """
        print("Extracting comprehensive feature set...")

        n_samples = fsr0.shape[0]
        all_features = []

        for i in range(n_samples):
            features = []

            # === FSR Features (both sensors) ===
            for fsr_data, name in [(fsr0[i], "FSR0"), (fsr1[i], "FSR1")]:
                # Normalize FSR data to [0,1] range based on sensor characteristics
                fsr_normalized = fsr_data / (
                    np.max(fsr_data) + 1e-8
                )  # Avoid division by zero

                # Global statistics on normalized data
                features.extend(
                    [
                        np.mean(fsr_normalized),
                        np.std(fsr_normalized),
                        np.max(fsr_normalized),  # Will be 1.0 after normalization
                        np.min(fsr_normalized),
                        np.percentile(fsr_normalized, 75),
                        np.percentile(fsr_normalized, 25),
                        np.sum(fsr_normalized > np.mean(fsr_normalized))
                        / fsr_normalized.size,  # Normalized active sensor ratio
                    ]
                )

                # Spatial features - center of pressure (normalized coordinates)
                spatial_mean = np.mean(fsr_data, axis=0)  # (16, 16)
                if np.sum(spatial_mean) > 0:
                    x_coords, y_coords = np.meshgrid(range(16), range(16))
                    cop_x = np.sum(x_coords * spatial_mean) / np.sum(spatial_mean)
                    cop_y = np.sum(y_coords * spatial_mean) / np.sum(spatial_mean)

                    # Normalize coordinates to [0,1] range
                    cop_x_norm = cop_x / 15.0  # 0-15 range -> 0-1
                    cop_y_norm = cop_y / 15.0
                    features.extend([cop_x_norm, cop_y_norm])

                    # Spatial spread (normalized)
                    spatial_var_x = np.sum(
                        ((x_coords - cop_x) ** 2) * spatial_mean
                    ) / np.sum(spatial_mean)
                    spatial_var_y = np.sum(
                        ((y_coords - cop_y) ** 2) * spatial_mean
                    ) / np.sum(spatial_mean)
                    spatial_var_x_norm = spatial_var_x / (15.0**2)  # Normalize variance
                    spatial_var_y_norm = spatial_var_y / (15.0**2)
                    features.extend([spatial_var_x_norm, spatial_var_y_norm])
                else:
                    features.extend([0.5, 0.5, 0.0, 0.0])  # Center position as default

                # Temporal dynamics (normalized)
                temporal_signal = np.mean(
                    fsr_data.reshape(fsr_data.shape[0], -1), axis=1
                )
                if len(temporal_signal) > 10:
                    # Normalize temporal signal
                    signal_max = np.max(temporal_signal)
                    if signal_max > 0:
                        temporal_norm = temporal_signal / signal_max

                        # Basic temporal features on normalized signal
                        diff_signal = np.diff(temporal_norm)
                        features.extend(
                            [
                                np.mean(
                                    diff_signal
                                ),  # Mean rate of change (normalized)
                                np.std(diff_signal),  # Variability in change
                                np.max(temporal_norm)
                                - np.min(temporal_norm),  # Normalized range
                            ]
                        )
                    else:
                        features.extend([0.0, 0.0, 0.0])

                    # Basic spectral features (normalized) - reduced for FSR
                    if len(temporal_signal) > 32:
                        fft_vals = np.abs(fft(temporal_signal))
                        # Normalize FFT values
                        fft_norm = fft_vals / (np.max(fft_vals) + 1e-8)
                        features.extend(
                            [
                                np.mean(
                                    fft_norm[:20]
                                ),  # Low frequency content (normalized)
                                np.std(fft_norm[:20]),
                                np.argmax(fft_vals[:50])
                                / 50.0,  # Dominant frequency (normalized)
                            ]
                        )
                    else:
                        features.extend([0, 0, 0])
                else:
                    features.extend([0, 0, 0, 0, 0, 0])

            # === Accelerometer Features (normalized) ===
            acc_data = acc[i]
            magnitude = np.sqrt(np.sum(acc_data**2, axis=2))  # (1111, 16)

            # Normalize magnitude by its maximum value
            mag_max = np.max(magnitude)
            if mag_max > 0:
                magnitude_norm = magnitude / mag_max
            else:
                magnitude_norm = magnitude

            # Global magnitude statistics (normalized)
            features.extend(
                [
                    np.mean(magnitude_norm),
                    np.std(magnitude_norm),
                    np.max(magnitude_norm),  # Will be 1.0 after normalization
                    np.min(magnitude_norm),
                    np.percentile(magnitude_norm, 75),
                ]
            )

            # Per-axis global features (normalized)
            for axis in range(3):
                axis_data = acc_data[:, :, axis]
                axis_max = np.max(np.abs(axis_data))
                if axis_max > 0:
                    axis_norm = axis_data / axis_max
                    features.extend(
                        [
                            np.mean(axis_norm),
                            np.std(axis_norm),
                            (np.max(axis_norm) - np.min(axis_norm)),  # Normalized range
                        ]
                    )
                else:
                    features.extend([0.0, 0.0, 0.0])

            # Enhanced accelerometer temporal and spectral analysis
            magnitude_signal = np.mean(
                magnitude_norm, axis=1
            )  # Use normalized magnitude
            if len(magnitude_signal) > 10:
                diff_signal = np.diff(magnitude_signal)
                features.extend(
                    [
                        np.mean(
                            diff_signal
                        ),  # Already normalized due to input normalization
                        np.std(diff_signal),
                    ]
                )

                # Comprehensive spectrogram features for acceleration magnitude
                if len(magnitude_signal) > 32:
                    spec_features = self.extract_spectrogram_features_simple(
                        magnitude_signal
                    )
                    features.extend(spec_features)
                else:
                    features.extend([0, 0, 0, 0, 0])

                # Per-axis spectrogram analysis for accelerometer
                for axis in range(3):
                    axis_data = acc_data[:, :, axis]
                    axis_signal = np.mean(axis_data, axis=1)  # Average across sensors
                    
                    # Normalize axis signal
                    axis_max = np.max(np.abs(axis_signal))
                    if axis_max > 0:
                        axis_norm = axis_signal / axis_max
                        
                        if len(axis_norm) > 32:
                            # Detailed spectrogram for each axis
                            axis_spec_features = self.extract_spectrogram_features_simple(
                                axis_norm
                            )
                            features.extend(axis_spec_features)
                        else:
                            features.extend([0, 0, 0, 0, 0])
                    else:
                        features.extend([0, 0, 0, 0, 0])
            else:
                features.extend([0, 0] + [0] * 5 + [0] * (3 * 5))  # 2 diff + 5 magnitude spec + 15 axis specs

            # === Kinesthetic Features (normalized) ===
            kin_data = kinesthetic[i]

            # Normalize kinesthetic data by overall range to handle different coordinate systems
            kin_range = np.max(kin_data) - np.min(kin_data)
            if kin_range > 0:
                kin_normalized = (
                    kin_data - np.min(kin_data)
                ) / kin_range  # Normalize to [0,1]
            else:
                kin_normalized = kin_data

            # Basic statistics per axis (on normalized data)
            for axis in range(3):
                axis_data = kin_normalized[:, axis]
                features.extend(
                    [
                        np.mean(axis_data),
                        np.std(axis_data),
                        np.max(axis_data)
                        - np.min(axis_data),  # Range (will be in [0,1])
                        np.percentile(axis_data, 75)
                        - np.percentile(axis_data, 25),  # IQR (normalized)
                    ]
                )

            # Motion characteristics (normalized)
            if kin_data.shape[0] > 1:
                # Use normalized kinesthetic data for motion calculations
                displacement = np.sqrt(
                    np.sum((kin_normalized[-1] - kin_normalized[0]) ** 2)
                )
                velocity = np.diff(kin_normalized, axis=0)
                path_length = np.sum(np.sqrt(np.sum(velocity**2, axis=1)))

                features.extend(
                    [
                        displacement,  # Already normalized due to input normalization
                        path_length,  # Already normalized due to input normalization
                        (
                            path_length / len(kin_data) if len(kin_data) > 0 else 0
                        ),  # Average speed (normalized)
                    ]
                )

                # 3D motion complexity (normalized)
                for axis in range(3):
                    axis_vel = velocity[:, axis]
                    if len(axis_vel) > 5:
                        features.append(
                            np.std(axis_vel)
                        )  # Velocity variability per axis (normalized)
                    else:
                        features.append(0)
            else:
                features.extend([0, 0, 0, 0, 0, 0])  # 3 motion + 3 velocity features

            all_features.append(features)

        features_array = np.array(all_features)
        print(f"Extracted {features_array.shape[1]} comprehensive features per sample")

        return features_array

    def extract_spectrogram_features_simple(self, signal, fs=1.0):
        """
        Extract key spectrogram features without overwhelming detail
        """
        if len(signal) < 16:
            return [0, 0, 0, 0, 0]

        try:
            window_size = min(64, len(signal) // 4)
            overlap = window_size // 2

            frequencies, times, Sxx = spectrogram(
                signal,
                fs=fs,
                nperseg=window_size,
                noverlap=overlap,
                scaling="density",
                mode="magnitude",
            )

            if Sxx.size == 0:
                return [0, 0, 0, 0, 0]

            # Normalize spectrogram for consistent scaling
            Sxx_max = np.max(Sxx)
            if Sxx_max > 0:
                Sxx_norm = Sxx / Sxx_max
            else:
                Sxx_norm = Sxx

            # Key spectral statistics (normalized)
            features = [
                np.mean(Sxx_norm),  # Overall spectral power (normalized)
                np.std(Sxx_norm),  # Spectral variability (normalized)
                np.max(Sxx_norm),  # Peak power (will be 1.0 after normalization)
                (
                    np.mean(frequencies[np.argmax(Sxx, axis=0)]) / (fs / 2)
                    if fs > 0
                    else 0
                ),  # Normalized dominant frequency
                (
                    np.std(frequencies[np.argmax(Sxx, axis=0)]) / (fs / 2)
                    if fs > 0
                    else 0
                ),  # Normalized frequency stability
            ]

            return features

        except Exception:
            return [0, 0, 0, 0, 0]

    def extract_simple_features(self, fsr0, fsr1, acc, kinesthetic):
        """
        Extract simple robust feature set (from haptic_grouped_classifier.py)
        """
        print("Extracting simple robust feature set...")

        n_samples = fsr0.shape[0]
        all_features = []

        for i in range(n_samples):
            features = []

            # FSR0 - Global features only (normalized)
            fsr0_data = fsr0[i]
            fsr0_max = np.max(fsr0_data)
            if fsr0_max > 0:
                fsr0_norm = fsr0_data / fsr0_max
                features.extend(
                    [
                        np.mean(fsr0_norm),
                        np.std(fsr0_norm),
                        np.max(fsr0_norm),  # Will be 1.0
                        np.percentile(fsr0_norm, 75),
                        np.sum(fsr0_norm > np.mean(fsr0_norm))
                        / fsr0_norm.size,  # Normalized count
                    ]
                )
            else:
                features.extend([0, 0, 0, 0, 0])

            # FSR1 - Global features only (normalized)
            fsr1_data = fsr1[i]
            fsr1_max = np.max(fsr1_data)
            if fsr1_max > 0:
                fsr1_norm = fsr1_data / fsr1_max
                features.extend(
                    [
                        np.mean(fsr1_norm),
                        np.std(fsr1_norm),
                        np.max(fsr1_norm),  # Will be 1.0
                        np.percentile(fsr1_norm, 75),
                        np.sum(fsr1_norm > np.mean(fsr1_norm))
                        / fsr1_norm.size,  # Normalized count
                    ]
                )
            else:
                features.extend([0, 0, 0, 0, 0])

            # Accelerometer - Magnitude features (normalized)
            acc_data = acc[i]
            magnitude = np.sqrt(np.sum(acc_data**2, axis=2))
            mag_max = np.max(magnitude)
            if mag_max > 0:
                magnitude_norm = magnitude / mag_max
                temporal_change = np.diff(magnitude_norm.mean(axis=1))
                features.extend(
                    [
                        np.mean(magnitude_norm),
                        np.std(magnitude_norm),
                        np.max(magnitude_norm),  # Will be 1.0
                        np.mean(temporal_change) if len(temporal_change) > 0 else 0,
                    ]
                )
            else:
                features.extend([0, 0, 0, 0])

            # Kinesthetic - Motion features (normalized)
            kin_data = kinesthetic[i]

            # Normalize kinesthetic data
            kin_range = np.max(kin_data) - np.min(kin_data)
            if kin_range > 0:
                kin_norm = (kin_data - np.min(kin_data)) / kin_range
                displacement = np.sqrt(np.sum((kin_norm[-1] - kin_norm[0]) ** 2))

                if len(kin_norm) > 1:
                    path_length = np.sum(
                        np.sqrt(np.sum(np.diff(kin_norm, axis=0) ** 2, axis=1))
                    )
                else:
                    path_length = 0

                features.extend(
                    [
                        displacement,
                        path_length,
                        np.std(kin_norm[:, 0]),
                        np.std(kin_norm[:, 1]),
                        np.std(kin_norm[:, 2]),
                    ]
                )
            else:
                features.extend([0, 0, 0, 0, 0])

            all_features.append(features)

        features_array = np.array(all_features)
        print(f"Extracted {features_array.shape[1]} simple features per sample")

        return features_array

    def apply_dimensionality_reduction(self, X_train, X_test, target_components=50):
        """
        Apply PCA dimensionality reduction
        """
        print("Applying PCA dimensionality reduction...")
        print(f"Original feature dimensions: {X_train.shape[1]}")

        # Determine number of components
        n_components = min(target_components, X_train.shape[1], X_train.shape[0] - 1)

        # Apply PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        # Report variance explained
        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        print(f"PCA components: {n_components}")
        print(f"Variance explained: {cumvar[-1]:.3f}")
        print(f"Components for 95% variance: {np.argmax(cumvar >= 0.95) + 1}")

        return X_train_pca, X_test_pca

    def load_and_process_data(self):
        """
        Load data and prepare for multi-output classification
        """
        print("Loading processed haptic data...")

        if not os.path.exists(self.data_path):
            print(f"Data file not found: {self.data_path}")
            return False

        # Load data
        data = np.load(self.data_path)

        # Extract arrays
        fsr0 = data["fsr0"]
        fsr1 = data["fsr1"]
        acc = data["acc"]
        kinesthetic = data["kinesthetic"]
        labels = data["labels"]

        print("Data shapes:")
        print(f"  FSR0: {fsr0.shape}")
        print(f"  FSR1: {fsr1.shape}")
        print(f"  Accelerometer: {acc.shape}")
        print(f"  Kinesthetic: {kinesthetic.shape}")
        print(f"  Labels: {labels.shape}")

        # Extract features based on mode
        if self.feature_mode == "comprehensive":
            self.X = self.extract_comprehensive_features(fsr0, fsr1, acc, kinesthetic)
        else:
            self.X = self.extract_simple_features(fsr0, fsr1, acc, kinesthetic)

        # Prepare labels for each factor
        self.y_letter = labels[:, 0]
        self.y_stiffness = labels[:, 1]
        self.y_shape = labels[:, 2]
        self.y_frequency = labels[:, 3]

        # Create object classification (15 unique combinations of letter, stiffness, shape)
        object_keys = [
            tuple(label[:3]) for label in labels
        ]  # [('C', 'hard', 'convex'), ...]
        unique_object_keys = sorted(set(object_keys))
        object_map = {
            key: idx for idx, key in enumerate(unique_object_keys)
        }  # maps to 0–14
        self.y_object = np.array(
            [object_map[tuple(label[:3])] for label in labels]
        )  # shape (450,)

        # Create action labels (frequency converted to integer)
        self.y_action = np.array(
            [int(float(label[3]) * 10) for label in labels]
        )  # shape (450,)

        # Store object mapping for interpretation
        self.object_keys = unique_object_keys
        self.object_map = object_map

        # Create label encoders for individual factors (keeping for compatibility)
        for factor in ["letter", "stiffness", "shape", "frequency"]:
            self.label_encoders[factor] = LabelEncoder()

        # Fit and transform individual factor labels
        self.y_letter = self.label_encoders["letter"].fit_transform(self.y_letter)
        self.y_stiffness = self.label_encoders["stiffness"].fit_transform(
            self.y_stiffness
        )
        self.y_shape = self.label_encoders["shape"].fit_transform(self.y_shape)
        self.y_frequency = self.label_encoders["frequency"].fit_transform(
            self.y_frequency
        )

        # Show distributions
        print("\nLabel distributions:")
        for name, encoder, y in [
            ("Letter", self.label_encoders["letter"], self.y_letter),
            ("Stiffness", self.label_encoders["stiffness"], self.y_stiffness),
            ("Shape", self.label_encoders["shape"], self.y_shape),
            ("Frequency", self.label_encoders["frequency"], self.y_frequency),
        ]:
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n{name}:")
            for val, count in zip(encoder.inverse_transform(unique), counts):
                print(f"  {val}: {count} samples")

        # Show new object and action distributions
        unique_objects, object_counts = np.unique(self.y_object, return_counts=True)
        print(
            f"\nObject combinations (letter-stiffness-shape): {len(unique_objects)} unique"
        )
        print(
            f"  Samples per object: min={min(object_counts)}, max={max(object_counts)}, mean={np.mean(object_counts):.1f}"
        )

        unique_actions, action_counts = np.unique(self.y_action, return_counts=True)
        print(f"\nAction frequencies: {len(unique_actions)} unique")
        for action, count in zip(unique_actions, action_counts):
            freq_value = action / 10.0
            print(f"  {freq_value:.1f} Hz: {count} samples")

        return True

    def train_models(
        self, test_size=0.2, random_state=42, use_pca=True, use_ensemble=True
    ):
        """
        Train models with optional PCA and ensemble methods
        """
        if self.X is None:
            print("No data available for training")
            return False

        # Split data - stratify by object classes for better representation
        indices = np.arange(len(self.X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_object,
        )

        X_train = self.X[train_idx]
        X_test = self.X[test_idx]

        # Apply dimensionality reduction if requested
        if use_pca and X_train.shape[1] > 50:
            X_train_proc, X_test_proc = self.apply_dimensionality_reduction(
                X_train, X_test
            )
        else:
            X_train_proc, X_test_proc = X_train, X_test
            print(f"Using original features: {X_train.shape[1]}")

        self.X_train = X_train_proc
        self.X_test = X_test_proc

        # Store test labels
        self.test_labels = {
            "letter": self.y_letter[test_idx],
            "stiffness": self.y_stiffness[test_idx],
            "shape": self.y_shape[test_idx],
            "frequency": self.y_frequency[test_idx],
            "object": self.y_object[test_idx],
            "action": self.y_action[test_idx],
        }

        print("\nTraining models for each factor...")
        print("=" * 60)

        # Define base models
        base_models = {
            "rf": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=random_state,
            ),
            "svm": SVC(kernel="rbf", probability=True, random_state=random_state),
            "lr": LogisticRegression(random_state=random_state, max_iter=1000),
        }

        results = {}

        for name, y_full in [
            ("Letter", self.y_letter),
            ("Stiffness", self.y_stiffness),
            ("Shape", self.y_shape),
            ("Frequency", self.y_frequency),
            ("Object", self.y_object),
            ("Action", self.y_action),
        ]:
            print(f"\nTraining {name} classifiers...")

            y_train = y_full[train_idx]
            y_test = y_full[test_idx]

            factor_results = {}

            # Train individual models
            for model_name, base_model in base_models.items():
                model = Pipeline(
                    [("scaler", StandardScaler()), ("classifier", base_model)]
                )

                model.fit(self.X_train, y_train)

                train_score = model.score(self.X_train, y_train)
                test_score = model.score(self.X_test, y_test)

                factor_results[model_name] = {
                    "model": model,
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                }

                print(
                    f"  {model_name.upper()}: Train={train_score:.3f}, Test={test_score:.3f}"
                )

            # Create ensemble if requested
            if use_ensemble:
                ensemble = VotingClassifier(
                    estimators=[
                        (
                            "rf",
                            Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    ("classifier", base_models["rf"]),
                                ]
                            ),
                        ),
                        (
                            "svm",
                            Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    ("classifier", base_models["svm"]),
                                ]
                            ),
                        ),
                        (
                            "lr",
                            Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    ("classifier", base_models["lr"]),
                                ]
                            ),
                        ),
                    ],
                    voting="soft",
                )

                ensemble.fit(self.X_train, y_train)

                train_score = ensemble.score(self.X_train, y_train)
                test_score = ensemble.score(self.X_test, y_test)

                factor_results["ensemble"] = {
                    "model": ensemble,
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                }

                print(f"  ENSEMBLE: Train={train_score:.3f}, Test={test_score:.3f}")

            # Cross-validation for best individual model
            best_model_name = max(
                [k for k in factor_results.keys() if k != "ensemble"],
                key=lambda x: factor_results[x]["test_accuracy"],
            )
            best_model = factor_results[best_model_name]["model"]

            cv_scores = cross_val_score(
                best_model,
                np.vstack([self.X_train, self.X_test]),
                np.hstack([y_train, y_test]),
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
                scoring="accuracy",
            )

            factor_results["best_model"] = best_model_name
            factor_results["cv_mean"] = cv_scores.mean()
            factor_results["cv_std"] = cv_scores.std()

            print(
                f"  Best: {best_model_name.upper()}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )

            results[name.lower()] = factor_results

            # Store the best performing model for evaluation
            if use_ensemble and "ensemble" in factor_results:
                self.models[name.lower()] = factor_results["ensemble"]["model"]
            else:
                self.models[name.lower()] = best_model

        self.results = results
        return True

    def evaluate_combined_performance(self):
        """
        Evaluate combined multi-output performance
        """
        print("\n" + "=" * 60)
        print("COMBINED PREDICTION EVALUATION")
        print("=" * 60)

        # Get predictions for each factor
        predictions = {}
        for name in ["letter", "stiffness", "shape", "frequency"]:
            predictions[name] = self.models[name].predict(self.X_test)

        # Combine predictions into full condition labels
        predicted_conditions = []
        true_conditions = []

        for i in range(len(self.X_test)):
            pred_parts = []
            true_parts = []

            for name in ["letter", "stiffness", "shape", "frequency"]:
                pred_val = self.label_encoders[name].inverse_transform(
                    [predictions[name][i]]
                )[0]
                true_val = self.label_encoders[name].inverse_transform(
                    [self.test_labels[name][i]]
                )[0]
                pred_parts.append(str(pred_val))
                true_parts.append(str(true_val))

            predicted_conditions.append("_".join(pred_parts))
            true_conditions.append("_".join(true_parts))

        # Calculate combined accuracy
        combined_accuracy = np.mean(
            np.array(predicted_conditions) == np.array(true_conditions)
        )
        n_unique_conditions = len(np.unique(true_conditions))
        print(
            f"\nCombined {n_unique_conditions}-condition accuracy: {combined_accuracy:.3f}"
        )

        # Analyze partial matches
        partial_matches = {
            "letter_only": 0,
            "at_least_one": 0,
            "at_least_two": 0,
            "at_least_three": 0,
            "all_four": 0,
        }

        for i in range(len(self.X_test)):
            correct_count = 0

            if predictions["letter"][i] == self.test_labels["letter"][i]:
                correct_count += 1
                partial_matches["letter_only"] += 1
            if predictions["stiffness"][i] == self.test_labels["stiffness"][i]:
                correct_count += 1
            if predictions["shape"][i] == self.test_labels["shape"][i]:
                correct_count += 1
            if predictions["frequency"][i] == self.test_labels["frequency"][i]:
                correct_count += 1

            if correct_count >= 1:
                partial_matches["at_least_one"] += 1
            if correct_count >= 2:
                partial_matches["at_least_two"] += 1
            if correct_count >= 3:
                partial_matches["at_least_three"] += 1
            if correct_count == 4:
                partial_matches["all_four"] += 1

        n_test = len(self.X_test)
        print("\nPartial match analysis:")
        for key, count in partial_matches.items():
            print(
                f"  {key.replace('_', ' ').title()}: {count}/{n_test} ({count/n_test*100:.1f}%)"
            )

        return combined_accuracy

    def visualize_results(self):
        """
        Create comprehensive visualizations
        """
        if not self.models:
            return

        plt.style.use("default")
        sns.set_palette("husl")

        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)

        # 1. Individual factor performance
        ax1 = fig.add_subplot(gs[0, :2])
        factors = ["letter", "stiffness", "shape", "frequency"]

        # Get best performance for each factor
        train_accs = []
        test_accs = []
        cv_means = []
        for factor in factors:
            if factor in self.results:
                best_model_name = self.results[factor]["best_model"]
                train_accs.append(
                    self.results[factor][best_model_name]["train_accuracy"]
                )
                test_accs.append(self.results[factor][best_model_name]["test_accuracy"])
                cv_means.append(self.results[factor]["cv_mean"])
            else:
                train_accs.append(0)
                test_accs.append(0)
                cv_means.append(0)

        x = np.arange(len(factors))
        width = 0.25

        bars1 = ax1.bar(
            x - width, train_accs, width, label="Train", alpha=0.8, color="#3498db"
        )
        bars2 = ax1.bar(x, test_accs, width, label="Test", alpha=0.8, color="#e67e22")
        bars3 = ax1.bar(
            x + width, cv_means, width, label="CV Mean", alpha=0.8, color="#2ecc71"
        )

        ax1.set_xlabel("Experimental Factor")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Individual Factor Classification Performance")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f.capitalize() for f in factors])
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # 2. PCA variance explained (if PCA was used)
        ax2 = fig.add_subplot(gs[0, 2:4])
        if self.pca is not None:
            cumvar = np.cumsum(self.pca.explained_variance_ratio_)
            ax2.plot(range(1, len(cumvar) + 1), cumvar, "bo-", alpha=0.7)
            ax2.axhline(
                y=0.95, color="r", linestyle="--", alpha=0.7, label="95% variance"
            )
            ax2.set_xlabel("Number of Components")
            ax2.set_ylabel("Cumulative Variance Explained")
            ax2.set_title("PCA Variance Explained")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(
                0.5,
                0.5,
                "PCA not applied",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("PCA Analysis")

        # 3. Combined performance comparison
        ax3 = fig.add_subplot(gs[0, 4:6])
        combined_acc = self.evaluate_and_get_combined_accuracy()

        methods = ["Random\nGuess", "Letter\nOnly", "Multi-Output\nCombined"]
        accuracies = [
            1 / 45,  # Random guess for 45 actual conditions
            (
                test_accs[0] * (1 / 15) if test_accs[0] > 0 else 0
            ),  # Letter only (15 conditions per letter)
            combined_acc,
        ]
        colors = ["#95a5a6", "#f39c12", "#27ae60"]

        bars = ax3.bar(methods, accuracies, color=colors, alpha=0.8)
        ax3.set_title("45-Condition Performance")
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0, max(accuracies) * 1.2)
        ax3.grid(True, alpha=0.3)

        for bar, acc in zip(bars, accuracies):
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.002,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Main 15x15 Object Classification Confusion Matrix
        ax4 = fig.add_subplot(gs[1, :2])
        if 'object' in self.models:
            object_model = self.models['object']
            y_pred = object_model.predict(self.X_test)
            y_true = self.test_labels['object']
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize confusion matrix by row (true labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            im = ax4.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
            ax4.figure.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            
            # Create object class labels
            classes = [f"{key[0]}-{key[1]}-{key[2]}" for key in self.object_keys]
            ax4.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes,
                   yticklabels=classes,
                   title="15×15 Object Classification (Letter-Stiffness-Shape)",
                   ylabel="Actual",
                   xlabel="Predicted")
            
            plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax4.get_yticklabels(), rotation=0)
            
            # Add text annotations with smaller font for 15x15
            thresh = 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = f"{cm_normalized[i, j]:.2f}" if cm_normalized[i, j] > 0.01 else ""
                    ax4.text(j, i, text, ha="center", va="center",
                           color="white" if cm_normalized[i, j] > thresh else "black",
                           fontsize=6, fontweight="bold")
        
        # 5-8. Smaller confusion matrices for individual factors  
        factor_models = [(name, model) for name, model in self.models.items() 
                        if name in ['letter', 'stiffness', 'shape', 'frequency']]
        
        for idx, (name, model) in enumerate(factor_models):
            row = 1 + (idx + 2) // 3  # Start from position 2 in grid
            col = (idx + 2) % 3
            ax = fig.add_subplot(gs[row, col])

            y_pred = model.predict(self.X_test)
            y_true = self.test_labels[name]
            cm = confusion_matrix(y_true, y_pred)

            # Normalize confusion matrix by row (true labels)
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Handle any NaN values (in case of zero rows)
            cm_normalized = np.nan_to_num(cm_normalized)

            im = ax.imshow(
                cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Use label encoders for individual factors
            classes = self.label_encoders[name.lower()].classes_
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=classes,
                yticklabels=classes,
                title=f"{name.capitalize()} Confusion Matrix (Normalized)",
                ylabel="Actual",
                xlabel="Predicted",
            )

            if len(classes[0]) > 2:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Add text annotations with normalized values
            thresh = 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = f"{cm_normalized[i, j]:.2f}"
                    ax.text(j, i, text, ha="center", va="center",
                           color="white" if cm_normalized[i, j] > thresh else "black",
                           fontsize=10, fontweight="bold")

        plt.suptitle(
            f"Haptic Classification Results ({self.feature_mode.title()} Features)",
            fontsize=16,
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig("results.png", dpi=300, bbox_inches="tight")
        plt.show()

    def evaluate_and_get_combined_accuracy(self):
        """Helper to get combined accuracy"""
        predictions = {}
        for name in ["letter", "stiffness", "shape", "frequency"]:
            predictions[name] = self.models[name].predict(self.X_test)

        predicted_conditions = []
        true_conditions = []

        for i in range(len(self.X_test)):
            pred_parts = []
            true_parts = []

            for name in ["letter", "stiffness", "shape", "frequency"]:
                pred_val = self.label_encoders[name].inverse_transform(
                    [predictions[name][i]]
                )[0]
                true_val = self.label_encoders[name].inverse_transform(
                    [self.test_labels[name][i]]
                )[0]
                pred_parts.append(str(pred_val))
                true_parts.append(str(true_val))

            predicted_conditions.append("_".join(pred_parts))
            true_conditions.append("_".join(true_parts))

        return np.mean(np.array(predicted_conditions) == np.array(true_conditions))

    def analyze_feature_importance(self):
        """
        Analyze feature importance using mutual information
        """
        if self.X is None or not self.models:
            print("No trained models available.")
            return None

        print("\nFeature Importance Analysis:")
        print("=" * 50)

        X = self.X  # Use original features for interpretability
        feature_importance = {}

        for factor in ["letter", "stiffness", "shape", "frequency"]:
            y_factor = getattr(self, f"y_{factor}")

            # Mutual information scores
            mi_scores = mutual_info_classif(X, y_factor, random_state=42)

            # Get top 10 most important features
            top_indices = np.argsort(mi_scores)[-10:]

            feature_importance[factor] = {
                "mi_scores": mi_scores,
                "top_features": top_indices,
                "top_scores": mi_scores[top_indices],
            }

            print(f"\n{factor.upper()} - Top 10 Features:")
            for i, (idx, score) in enumerate(
                zip(top_indices[::-1], mi_scores[top_indices][::-1])
            ):
                print(f"  {i+1}. Feature {idx}: {score:.4f}")

        return feature_importance

    def visualize_embeddings(self):
        """
        Create t-SNE and UMAP visualizations
        """
        if self.X is None:
            print("No data available.")
            return None

        print("Creating dimensionality reduction visualizations...")

        X_viz = self.X_reduced if self.X_reduced is not None else self.X

        # Create embeddings
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_viz) - 1))
        tsne_embedding = tsne.fit_transform(X_viz)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Dimensionality Reduction Analysis", fontsize=16)

        factors = ["letter", "stiffness", "shape", "frequency"]
        colors = ["viridis", "plasma", "inferno", "cividis"]

        for i, (factor, cmap) in enumerate(zip(factors, colors)):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            y_factor = getattr(self, f"y_{factor}")
            classes = self.label_encoders[factor].classes_

            scatter = ax.scatter(
                tsne_embedding[:, 0],
                tsne_embedding[:, 1],
                c=y_factor,
                cmap=cmap,
                alpha=0.7,
                s=30,
            )
            ax.set_title(f"t-SNE: {factor.capitalize()}")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_ticks(range(len(classes)))
            cbar.set_ticklabels(classes)

        plt.tight_layout()
        plt.savefig("embeddings.png", dpi=300, bbox_inches="tight")
        plt.show()

        return tsne_embedding

    def run_full_pipeline(self, use_pca=True, use_ensemble=True, analyze_features=True):
        """
        Run the complete classification pipeline
        """
        print("Starting Unified Haptic Classification Pipeline")
        print(f"Feature mode: {self.feature_mode}")
        print(f"PCA: {'Enabled' if use_pca else 'Disabled'}")
        print(f"Ensemble: {'Enabled' if use_ensemble else 'Disabled'}")
        print(f"Feature Analysis: {'Enabled' if analyze_features else 'Disabled'}")
        print("=" * 60)

        # Load and process data
        if not self.load_and_process_data():
            print("Failed to load data")
            return False

        # Train models
        if not self.train_models(use_pca=use_pca, use_ensemble=use_ensemble):
            print("Failed to train models")
            return False

        # Evaluate combined performance
        combined_accuracy = self.evaluate_combined_performance()

        # Visualize results
        self.visualize_results()

        # Optional additional analysis
        if analyze_features:
            self.analyze_feature_importance()
            self.visualize_embeddings()

        print("\nPipeline completed successfully!")
        print(f"Final combined 45-condition accuracy: {combined_accuracy:.3f}")

        return True


# Usage examples
if __name__ == "__main__":
    print("=== Haptic Classification with Comprehensive Features ===")
    classifier_comprehensive = HapticClassifier(feature_mode="comprehensive")
    classifier_comprehensive.run_full_pipeline(use_pca=True, use_ensemble=True)

