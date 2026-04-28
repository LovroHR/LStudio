"""
5. laboratorijska vježba - poboljšani obični CNN bez transfer learninga

Dataset:
Flowers Dataset - Kaggle
https://www.kaggle.com/datasets/imsparsh/flowers-dataset

Struktura foldera:
Lab5/
    poboljsani_cnn_lab5.py
    dataset/
        daisy/
        dandelion/
        rose/
        sunflower/
        tulip/

Pokretanje:
python poboljsani_cnn_lab5.py

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
EPOCHS = 20
SEED = 123

print("Postoji li folder:", os.path.exists(DATA_DIR))

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

# Dio validation skupa koristimo kao testni skup
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# Augmentacija - pomaže da model bolje generalizira
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.10),
])

# Poboljšani CNN
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),

    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.20),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.30),

    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.35),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.50),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Ljepši TensorBoard logovi - svaki run ide u poseban folder
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "najbolji_cnn_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[tensorboard, early_stop, checkpoint]
)

best_model = tf.keras.models.load_model("najbolji_cnn_model.keras")

loss, acc = best_model.evaluate(test_ds)
print("Test accuracy:", acc)

best_model.save("poboljsani_cnn_model.keras")
print("Model je spremljen kao poboljsani_cnn_model.keras")
print("TensorBoard logovi su spremljeni u:", log_dir)
