"""
5. labos - optimizirani CNN od nule za klasifikaciju cvijeća

VAŽNO:
Ovaj kod NE koristi transfer learning i NE koristi gotovu mrežu poput MobileNetV2.
Model se trenira od nule, ali je optimiziran za bolji rezultat.

Dataset:
Flowers Dataset - Kaggle
https://www.kaggle.com/datasets/imsparsh/flowers-dataset

Očekivana struktura:
Lab5/
    cnn_od_nule_optimiziran.py
    dataset/
        daisy/
        dandelion/
        rose/
        sunflower/
        tulip/

Pokretanje:
python cnn_od_nule_optimiziran.py

TensorBoard:
tensorboard --logdir logs
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ---------------------------------------------------------
# OSNOVNE POSTAVKE
# ---------------------------------------------------------
DATA_DIR = "dataset"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 123

print("Postoji li dataset folder:", os.path.exists(DATA_DIR))
print("GPU uređaji:", tf.config.list_physical_devices("GPU"))

# Ako TensorFlow vidi GPU, koristi memory growth da ne zauzme svu memoriju odmah
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ---------------------------------------------------------
# UČITAVANJE DATASETA
# ---------------------------------------------------------
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

# Pola validation skupa ostaje validation, pola koristimo kao test
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ---------------------------------------------------------
# AUGMENTACIJA
# ---------------------------------------------------------
# Ovo pomaže jer model vidi različite verzije iste slike
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.12),
    layers.RandomTranslation(0.08, 0.08),
], name="augmentacija")

# ---------------------------------------------------------
# CNN OD NULE
# ---------------------------------------------------------
# regularizer smanjuje overfitting
reg = regularizers.l2(0.0005)

model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),

    data_augmentation,
    layers.Rescaling(1./255),

    # BLOK 1
    layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(),
    layers.Dropout(0.15),

    # BLOK 2
    layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(),
    layers.Dropout(0.20),

    # BLOK 3
    layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # BLOK 4
    layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D(),
    layers.Dropout(0.30),

    # BLOK 5
    layers.Conv2D(512, 3, padding="same", kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, kernel_regularizer=reg),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.45),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------
# TENSORBOARD I CALLBACKOVI
# ---------------------------------------------------------
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

# Zaustavi treniranje kad se accuracy prestane poboljšavati
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True
)

# Smanji learning rate kad model zapne
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00003,
    verbose=1
)

# Spremi najbolji model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "najbolji_cnn_od_nule.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ---------------------------------------------------------
# TRENIRANJE
# ---------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[tensorboard, early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ---------------------------------------------------------
# TESTIRANJE
# ---------------------------------------------------------
best_model = tf.keras.models.load_model("najbolji_cnn_od_nule.keras")

test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)

print("\nTest loss:", test_loss)
print("Test accuracy:", test_acc)

best_model.save("cnn_od_nule_optimiziran_model.keras")

print("\nModel spremljen kao: cnn_od_nule_optimiziran_model.keras")
print("Najbolji model spremljen kao: najbolji_cnn_od_nule.keras")
print("TensorBoard logovi:", log_dir)
