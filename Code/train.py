# ====== train.py (Fast training, Windows-safe) ======
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
TRAIN_DIR = BASE_DIR / "Dataset" / "Train"
VAL_DIR   = BASE_DIR / "Dataset" /  "Validation"
MODEL_OUT = BASE_DIR / "deepfake_detection_model.h5"

# ---- Hyperparams ----
BATCH = 64
EPOCHS = 1
STEPS = 150
VAL_STEPS = 60

def build_and_train():
    print("Train dir :", TRAIN_DIR)
    print("Val dir   :", VAL_DIR)

    # Data generators
    train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    val_gen   = ImageDataGenerator(rescale=1./255)

    train = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=(96, 96), batch_size=BATCH, class_mode="binary")

    val = val_gen.flow_from_directory(
        VAL_DIR, target_size=(96, 96), batch_size=BATCH, class_mode="binary")

    # Model
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Callbacks
    ckpt = ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1)
    es = EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)

    # IMPORTANT: workers=1 & multiprocessing False (WINDOWS FIX)
    history = model.fit(
        train,
        epochs=EPOCHS,
        steps_per_epoch=STEPS,
        validation_data=val,
        validation_steps=VAL_STEPS,
        workers=1,
        use_multiprocessing=False,
        callbacks=[ckpt, es]
    )

    print("\n✅ Model training finished!")
    print(f"✅ Model saved at: {MODEL_OUT}")

if __name__ == "__main__":
    build_and_train()
