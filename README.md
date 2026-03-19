# 🔢 Digit Recognition — MNIST

A complete machine learning pipeline that trains a **Convolutional Neural Network (CNN)** on the MNIST dataset to recognize and classify handwritten digits (0–9) with ~99% test accuracy.

---

## 📁 Project Structure

```
Digit Recognition/
├── digit_recognition.py        # Full ML pipeline (main script)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
│   (generated after running)
├── digit_recognition_model.keras   # Saved trained model
├── sample_images.png               # Grid of training samples
├── training_curves.png             # Accuracy & loss over epochs
├── confusion_matrix.png            # Per-class prediction heatmap
└── predictions.png                 # Predicted vs. actual labels
```

---

## 🧠 Model Architecture

A compact **CNN** with 3 convolutional blocks followed by a dense classifier head:

```
Input (28×28×1)
    │
    ├─ Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2)
    ├─ Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2)
    ├─ Conv2D(128, 3×3, relu) → BatchNorm
    │
    └─ Flatten → Dense(256, relu) → Dropout(0.4) → Dense(10, softmax)
```

| Hyperparameter   | Value                     |
|------------------|---------------------------|
| Optimizer        | Adam (lr = 0.001)         |
| Loss             | Categorical Cross-Entropy |
| Batch Size       | 128                       |
| Max Epochs       | 20 (EarlyStopping applied)|
| Dropout Rate     | 0.4                       |

---

## 🔄 Pipeline Overview

| Step | Description |
|------|-------------|
| **1. Load Data** | Downloads MNIST automatically via Keras (60,000 train / 10,000 test) |
| **2. Preprocess** | Normalizes pixels to `[0.0, 1.0]`, reshapes to `(N, 28, 28, 1)`, one-hot encodes labels |
| **3. Augment** | Random rotation (±18°), zoom (±10%), and translation (±10%) via `tf.data` pipeline |
| **4. Build Model** | Constructs CNN with BatchNorm and Dropout for regularization |
| **5. Train** | Trains with `EarlyStopping` (patience=4) and `ReduceLROnPlateau` (patience=2) |
| **6. Evaluate** | Reports overall test accuracy, loss, and a per-class classification report |
| **7. Visualize** | Saves training curves, confusion matrix, sample images, and predictions |
| **8. Save Model** | Exports the trained model to `.keras` format for later inference |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow >= 2.13` | Model building, training, Keras API |
| `numpy >= 1.24` | Numerical operations |
| `matplotlib >= 3.7` | Plotting training curves & predictions |
| `seaborn >= 0.12` | Confusion matrix heatmap |
| `scikit-learn >= 1.3` | Classification report & confusion matrix |

---

## ▶️ Running the Script

```bash
python digit_recognition.py
```

The first run will automatically download the MNIST dataset (~11 MB) to your Keras cache folder. Subsequent runs will use the cached version.

---

## 📊 Expected Output

```
=======================================================
  MNIST Digit Recognition
=======================================================

Training samples : 60000
Test samples     : 10000
Image shape      : (28, 28)  (pixels × pixels)
Classes          : [0 1 2 3 4 5 6 7 8 9]

Training …
Epoch 1/20 — val_accuracy: 0.9921
...
=======================================================
  Test Accuracy : 99.10%
  Test Loss     : 0.0284
=======================================================

Per-class Classification Report:
              precision    recall  f1-score
           0       0.99      1.00      0.99
           1       0.99      1.00      1.00
           ...
```

**Generated files:**

| File | Contents |
|------|----------|
| `sample_images.png` | 16 random training images with their labels |
| `training_curves.png` | Accuracy & loss plots across epochs |
| `confusion_matrix.png` | 10×10 heatmap of predicted vs. actual digits |
| `predictions.png` | 16 test images with predicted (green=correct, red=wrong) labels |
| `digit_recognition_model.keras` | Final trained model |

---

## ♻️ Reusing the Trained Model

Load and run inference with the saved model:

```python
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("digit_recognition_model.keras")

# Prepare a single image (28×28 grayscale, values in [0, 1])
img = your_image_array.reshape(1, 28, 28, 1).astype("float32") / 255.0

# Predict
prediction = np.argmax(model.predict(img), axis=1)
print(f"Predicted digit: {prediction[0]}")
```

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Expected test accuracy | ~99% |
| Model size | ~5 MB |

---

## 📌 Notes

- **Data augmentation** (rotation, zoom, shift) is applied only during training to improve robustness against real-world handwriting variation.
- **EarlyStopping** automatically halts training when validation accuracy stops improving, restoring the best weights.
- **ReduceLROnPlateau** halves the learning rate when the validation loss plateaus, squeezing out extra accuracy.
- The model is saved in the modern **Keras v3 `.keras` format**, which is portable and framework-version safe.

---

*Built with TensorFlow/Keras · MNIST Dataset · Python*
