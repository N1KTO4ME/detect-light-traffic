import os
import json
import sqlite3
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

import cv2

logging.basicConfig(level=logging.INFO, filename="logs/app.log",
                    filemode="a",
                    format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# Параметры по умолчанию
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_model.h5")
CLASS_MAP_PATH = os.path.join(MODEL_DIR, "class_names.json")
OUTPUTS_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


# -------------------------
# 1) Подготовка датасета
def prepare_datasets(dataset_dir="dataset", img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=123):
    """
    Ожидается структура dataset/train/{class}/ и dataset/val/{class}/
    Возвращает train_ds, val_ds, class_names
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    logging.info(f"Preparing datasets from {train_dir} and {val_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed
    )

    class_names = train_ds.class_names
    logging.info(f"Found classes: {class_names}")

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Normalize datasets on the fly in model (or here)
    normalization_layer = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, class_names


# -------------------------
# 2) Модель (с нуля)
def build_model(input_shape=IMG_SIZE + (3,), num_classes=3):
    """
    Простая CNN, обучаемая с нуля.
    Не использует предобученные веса.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# -------------------------
# 3) Тренировка, сохранение графиков и confusion matrix
def train_model(train_ds, val_ds, class_names, epochs=20, model_path=MODEL_PATH, outputs_dir=OUTPUTS_DIR):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    logging.info(f"Starting training for {epochs} epochs")

    num_classes = len(class_names)
    model = build_model(num_classes=num_classes)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Сохраняем модель
    model.save(model_path)
    logging.info(f"Saved model to {model_path}")

    # Сохраняем class names
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)
    logging.info(f"Saved class map to {CLASS_MAP_PATH}")

    # Графики loss/accuracy
    plot_history(history, outputs_dir)

    # Confusion matrix на валидации (получим предсказания)
    save_confusion_matrix(model, val_ds, class_names, outputs_dir)

    return model, history


def plot_history(history, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)
    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(outputs_dir, "loss.png"))
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(outputs_dir, "accuracy.png"))
    plt.close()

    logging.info(f"Saved training plots to {outputs_dir}")


def save_confusion_matrix(model, val_ds, class_names, outputs_dir):
    # Соберём все метки и предсказания
    y_true = []
    y_pred = []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        preds_labels = np.argmax(preds, axis=1)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds_labels.tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info(f"Saved confusion matrix to {path}")

    # Also print classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with op
