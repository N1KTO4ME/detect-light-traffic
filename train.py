import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- 1. Загрузка датасета ---
img_size = (64, 64)
batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/val",
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)
class_names = train_ds.class_names
print("Классы:", class_names)

# --- 2. Аугментация данных ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomBrightness(0.3),
    tf.keras.layers.RandomContrast(0.3),
])
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# --- 3. Создание CNN модели ---
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),  # Дополнительный слой
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),  # Увеличили количество нейронов
    tf.keras.layers.Dropout(0.5),  # Регуляризация
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- 4. Обучение ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # Увеличили количество эпох
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
    ]
)

# --- 5. Визуализация обучения ---
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.title("Accuracy")
plt.savefig("accuracy.png")

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.title("Loss")
plt.savefig("loss.png")

# --- 6. Confusion Matrix ---
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix.png")

# --- 7. Сохранение модели ---
model.save("model.h5")
print("✅ Модель сохранена в model.h5")
