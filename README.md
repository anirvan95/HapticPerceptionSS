# HapticPerceptionSS

Code base for the Haptic Perception Workshop.

> ⚙️ **Python Requirement:**  
> Python version **3.9.13** is recommended. Please update requirements.txt if another version is installed.

---

## 🧪 Test Scripts

- `test_eskin.py`  
  Checks connectivity and prints mean FSR and accelerometer values.

- `test_realsense.py`  
  Checks RealSense camera connectivity and plots RGB and depth channels.

- `test_aruco_detection.py`  
  Detects ArUco markers in the RGB feed. Can be used to determine the camera's pose or location.

---

## 📊 Plot Scripts

- `plot_kinesthetic_data.py`  
  Plots the relative 6D pose (translation + rotation) of the e-skin/tactile sensor.

- `plot_tactile_data.py`  
  Real-time visualization of FSR and accelerometer values.

- `plot_tactile_data_3D.py`  
  Efficient 3D visualization of tactile data — intuitive for spatiotemporal inspection.

---

## 🧵 Main Scripts

- `record_haptic_data.py`  
  Records both **tactile** and **kinesthetic** data streams (at different sampling rates)  
  and saves them as `.npy` files for further analysis.

- `plot_recorded_data.py`  
  Visualizes recorded tactile data (mean FSR, mean ACC) and kinesthetic data (position).  
  Useful for evaluating experiment protocol, filtering steps, and pilot study consistency.

- `haptic_time_series_classifier.py`  
  **Advanced time-series classifier** for haptic perception data. Classifies letters (C, D, Q) using comprehensive feature extraction from multi-modal sensor data and machine learning models.

---

## 🤖 Machine Learning Classification

### `haptic_time_series_classifier.py`

**Advanced time-series classifier** for haptic perception data that classifies letters (C, D, Q) using multi-modal sensor fusion and comprehensive feature extraction.

#### **📊 Data Processing**
- **Input**: Processed dataset (`dataset/processed_data.npz`) containing 450 samples
- **Data Structure**: 
  - FSR sensors: Two 16×16 arrays over 1111 timesteps  
  - Accelerometer: 16×3 sensor array over 1111 timesteps
  - Kinesthetic: 3D pose data over 50 timesteps
  - Labels: Letter classification with metadata (stiffness, shape, frequency)

#### **🔧 Feature Extraction Pipeline**

**FSR Features (Force-Sensitive Resistors)**
- **Statistical**: Mean, std, max, min per sensor across time (256 sensors × 4 stats = 1024 features per FSR)
- **Global**: Overall statistics across entire sensor array (4 features per FSR)
- **Spatial Patterns**: Time-averaged spatial distribution, row/column sums (256 + 16 + 16 = 288 features per FSR)
- **Temporal Dynamics**: FFT analysis of global signal over time (3 frequency domain features per FSR)
- **Total per FSR**: ~1,319 features × 2 FSR arrays = **~2,638 FSR features**

**Accelerometer Features**
- **Statistical**: Mean, std, max, min per sensor-axis across time (48 channels × 4 stats = 192 features)
- **Global**: Overall statistics (4 features)
- **Per-Axis**: Statistics for X, Y, Z axes separately (3 axes × 4 stats = 12 features)
- **Magnitude**: 3D acceleration magnitude statistics (4 features)
- **Total**: **~212 accelerometer features**

**Kinesthetic Features (6DOF Pose)**
- **Per-Axis Statistical**: Mean, std, max, min, median for each spatial dimension (3 axes × 5 stats = 15 features)
- **Global**: Overall pose statistics (4 features)
- **Motion Dynamics**: Velocity statistics and total path length (5 features)
- **Total**: **~24 kinesthetic features**

**Combined Feature Vector**: **~2,874 features per sample**

#### **📋 Feature Summary Table**

| **Sensor Modality** | **Feature Category** | **Description** | **Dimensions** | **Feature Count** |
|:---------------------|:---------------------|:----------------|:---------------|:------------------|
| **FSR0 & FSR1** | Statistical | Mean, std, max, min per sensor across time | 256 sensors × 4 stats × 2 FSR | **2,048** |
| | Global | Overall array statistics | 4 stats × 2 FSR | **8** |
| | Spatial Patterns | Time-averaged distribution, row/column sums | (256 + 16 + 16) × 2 FSR | **576** |
| | Temporal Dynamics | FFT analysis of global signal | 3 freq features × 2 FSR | **6** |
| | **FSR Subtotal** | | | **2,638** |
| **Accelerometer** | Statistical | Mean, std, max, min per sensor-axis across time | 48 channels × 4 stats | **192** |
| | Global | Overall statistics | 4 stats | **4** |
| | Per-Axis | X, Y, Z axis statistics | 3 axes × 4 stats | **12** |
| | Magnitude | 3D acceleration magnitude statistics | 4 stats | **4** |
| | **Accelerometer Subtotal** | | | **212** |
| **Kinesthetic** | Per-Axis Statistical | Mean, std, max, min, median per dimension | 3 axes × 5 stats | **15** |
| | Global | Overall pose statistics | 4 stats | **4** |
| | Motion Dynamics | Velocity statistics + path length | 4 velocity + 1 path | **5** |
| | **Kinesthetic Subtotal** | | | **24** |
| | | | **TOTAL FEATURES** | **~2,874** |

#### **🧠 Machine Learning Models**

**Model Architecture**
- **Pipeline**: StandardScaler → Classifier for each model
- **Models Evaluated**:
  - **Random Forest**: 100 estimators, robust to feature noise
  - **Gradient Boosting**: 100 estimators, sequential learning
  - **Support Vector Machine**: RBF kernel, non-linear classification
  - **Logistic Regression**: Linear baseline with regularisation

**Evaluation Strategy**
- **Train/Test Split**: 80/20 stratified split
- **Cross-Validation**: 5-fold stratified CV for robust performance estimation
- **Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Model Selection**: Best model chosen by CV accuracy

#### **📈 Visualisation & Results**

**Generated Plots** (`haptic_classification_results.png`)
1. **Model Performance Comparison**: Train/test/CV accuracy bar chart
2. **Cross-Validation Distribution**: Boxplots showing model stability
3. **Class Distribution**: Balanced dataset visualization (150 samples per letter)
4. **Confusion Matrix**: Detailed classification performance for best model

**Expected Performance**
- **Dataset**: 450 samples (150 per letter: C, D, Q)
- **Feature Space**: High-dimensional (~2,874 features) from multi-modal sensor fusion
- **Classification**: 3-class balanced letter recognition task

#### **💻 Usage**

```python
# Basic usage
classifier = HapticTimeSeriesClassifier()
success, best_model_name, best_model = classifier.run_full_pipeline()

# Custom data path
classifier = HapticTimeSeriesClassifier(data_path="path/to/processed_data.npz")
classifier.run_full_pipeline()
```

**Dependencies**: `numpy`, `polars`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`

---

