"""
Optimalni CNN za klasifikaciju cvijeća + korištenje GPU-a ako je dostupan.

Dataset:
Flowers Dataset - Kaggle
https://www.kaggle.com/datasets/imsparsh/flowers-dataset

Struktura:
Lab5/
    optimalni_cnn_cvijece_gpu.py
    dataset/
        daisy/
        dandelion/
        rose/
        sunflower/
        tulip/

Pokretanje:
python optimalni_cnn_cvijece_gpu.py

TensorBoard:
tensorboard --logdir logs
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 123

print("Postoji li dataset folder:", os.path.exists(DATA_DIR))

# ------------------------------------------------------
# GPU POSTAVKE
# ------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
print("Pronađeni GPU uređaji:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.set_visible_devices(gpus[0], "GPU")
        DEVICE = "/GPU:0"

        print("Koristi se GPU:", gpus[0])

    except RuntimeError as e:
        print("Greška kod postavljanja GPU-a:", e)
        DEVICE = "/CPU:0"
else:
    DEVICE = "/CPU:0"
    print("GPU nije pronađen u TensorFlowu. Kod će raditi na CPU-u.")
    print("Ako inače imaš GPU, problem je u TensorFlow/CUDA instalaciji, ne u modelu.")

# ------------------------------------------------------
# DATASET
# ------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Klase:", class_names)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ------------------------------------------------------
# MODEL NA GPU-U AKO POSTOJI
# ------------------------------------------------------
with tf.device(DEVICE):

    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ])

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        data_augmentation,
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.15),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.20),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.30),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.45),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00005
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "najbolji_cvijece_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

with tf.device(DEVICE):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[tensorboard, early_stop, reduce_lr, checkpoint]
    )

best_model = tf.keras.models.load_model("najbolji_cvijece_model.keras")

with tf.device(DEVICE):
    test_loss, test_acc = best_model.evaluate(test_ds)

print("\nUređaj korišten za treniranje:", DEVICE)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

best_model.save("optimalni_cvijece_model.keras")

print("\nModel spremljen kao: optimalni_cvijece_model.keras")
print("TensorBoard logovi:", log_dir)
