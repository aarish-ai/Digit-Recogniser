"""
╔══════════════════════════════════════════════════════════════╗
║              DIGIT RECOGNITION — MNIST                       ║
║  Preprocess → Build CNN → Train → Evaluate → Predict         ║
╚══════════════════════════════════════════════════════════════╝

Dependencies:  pip install tensorflow matplotlib seaborn numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
# 1. LOAD & INSPECT DATA
# ──────────────────────────────────────────────
print("=" * 55)
print("  MNIST Digit Recognition")
print("=" * 55)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Image shape      : {X_train.shape[1:]}  (pixels × pixels)")
print(f"Classes          : {np.unique(y_train)}  (digits 0-9)\n")

# ──────────────────────────────────────────────
# 2. PREPROCESS
# ──────────────────────────────────────────────
# Normalise pixel values from [0, 255] to [0.0, 1.0]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# Reshape to (samples, 28, 28, 1) — CNN expects a channel dimension
X_train = X_train[..., np.newaxis]
X_test  = X_test[...,  np.newaxis]

# One-hot encode labels
NUM_CLASSES = 10
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"X_train shape after preprocessing: {X_train.shape}")
print(f"y_train shape after one-hot       : {y_train_oh.shape}\n")

# ──────────────────────────────────────────────
# 3. VISUALISE SAMPLE IMAGES
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
fig.suptitle("Sample MNIST Images", fontsize=14, fontweight="bold")
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i, ..., 0], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=120)
plt.show()
print("Saved → sample_images.png\n")

# ──────────────────────────────────────────────
# 4. BUILD CNN MODEL
# ──────────────────────────────────────────────
def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Compact CNN:
      Conv → MaxPool → Conv → MaxPool → Flatten → Dense → Dropout → Softmax
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape, name="conv1"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name="pool1"),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name="pool2"),

        # Block 3 — deeper features
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.BatchNormalization(),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu", name="fc1"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="DigitCNN")

    return model


model = build_model()
model.summary()

# ──────────────────────────────────────────────
# 5. COMPILE
# ──────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ──────────────────────────────────────────────
# 6. DATA AUGMENTATION (light — keeps digits recognisable)
# ──────────────────────────────────────────────
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.05),          # ±18°
    layers.RandomZoom(0.1),               # ±10% zoom
    layers.RandomTranslation(0.1, 0.1),   # ±10% shift
], name="augmentation")

# ──────────────────────────────────────────────
# 7. TRAIN
# ──────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS     = 20

# Build tf.data pipeline with augmentation
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
    .shuffle(60000)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (data_augmentation(x, training=True), y),
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test_oh))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True,
                  verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

print("\nTraining …")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)

# ──────────────────────────────────────────────
# 8. EVALUATE
# ──────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\n{'='*55}")
print(f"  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"{'='*55}\n")

# ──────────────────────────────────────────────
# 9. TRAINING CURVES
# ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Training History", fontsize=14, fontweight="bold")

ax1.plot(history.history["accuracy"],     label="Train Acc")
ax1.plot(history.history["val_accuracy"], label="Val Acc")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history["loss"],     label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=120)
plt.show()
print("Saved → training_curves.png\n")

# ──────────────────────────────────────────────
# 10. CONFUSION MATRIX
# ──────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()
print("Saved → confusion_matrix.png\n")

# Per-class report
print("Per-class Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=[str(i) for i in range(10)]))

# ──────────────────────────────────────────────
# 11. VISUALISE PREDICTIONS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
fig.suptitle("Predictions on Test Set", fontsize=14, fontweight="bold")

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i, ..., 0], cmap="gray")
    pred  = y_pred[i]
    true  = y_test[i]
    color = "green" if pred == true else "red"
    ax.set_title(f"P:{pred} T:{true}", fontsize=8, color=color)
    ax.axis("off")

plt.tight_layout()
plt.savefig("predictions.png", dpi=120)
plt.show()
print("Saved → predictions.png\n")

# ──────────────────────────────────────────────
# 12. SAVE MODEL
# ──────────────────────────────────────────────
model.save("digit_recognition_model.keras")
print("Model saved → digit_recognition_model.keras")
print("\nDone! ✓")
