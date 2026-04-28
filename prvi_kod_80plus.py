"""
Jednostavni CNN za 5. labos, podešen za bolji accuracy

Dataset:
Flowers Dataset - Kaggle
https://www.kaggle.com/datasets/imsparsh/flowers-dataset

Struktura:
Lab5/
    prvi_kod_80plus.py
    dataset/
        daisy/
        dandelion/
        rose/
        sunflower/
        tulip/

Pokretanje:
python prvi_kod_80plus.py

TensorBoard:
tensorboard --logdir logs
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "dataset"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 123

print("Postoji li folder:", os.path.exists(DATA_DIR))
print("GPU:", tf.config.list_physical_devices("GPU"))

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
print("Broj klasa:", num_classes)

# dio validation skupa koristimo kao test skup
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# Malo augmentacije da model bolje uči i ne zapamti slike napamet
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
])

model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),

    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.40),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    write_graph=True
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00005,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "najbolji_prvi_kod_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[tensorboard, early_stop, reduce_lr, checkpoint],
    verbose=1
)

best_model = tf.keras.models.load_model("najbolji_prvi_kod_model.keras")

loss, acc = best_model.evaluate(test_ds)

print("\nTest accuracy:", acc)

best_model.save("prvi_kod_80plus_model.keras")

print("\nModel spremljen kao: prvi_kod_80plus_model.keras")
print("TensorBoard logovi:", log_dir)
