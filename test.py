import cv2
import numpy as np
import tensorflow as tf
import os

# === Загрузка обученной модели ===
model = tf.keras.models.load_model("model.h5")
classes = ["green", "red", "yellow"]  # порядок как в dataset/train

# === Предсказание для ROI ===
def predict_signal(roi):
    try:
        img_resized = cv2.resize(roi, (64, 64))  # под размер входа в модель
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        pred = model.predict(img_array, verbose=0)
        label = classes[np.argmax(pred)]
        confidence = float(np.max(pred))
        return label, confidence
    except Exception:
        return None, 0.0

# === Поиск ламп внутри рамки ===
# def detect_signal_lights(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # поиск кругов (лампы светофора)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=8, maxRadius=60
    )

    results = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (cx, cy, r) in circles:
            lamp = roi[cy-r:cy+r, cx-r:cx+r]
            if lamp.size == 0:
                continue
            label, conf = predict_signal(lamp)
            if label is not None:
                results.append((label, conf, (cx, cy, r)))
    return results
def detect_signal_lights(roi):
    results = []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Детектим круги в маске для каждого цвета
    for color, (lower, upper) in {
        "red": ([0, 120, 70], [10, 255, 255]),
        "yellow": ([20, 100, 100], [30, 255, 255]),
        "green": ([40, 100, 100], [80, 255, 255])
    }.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        gray_masked = cv2.bitwise_and(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), mask)
        circles = cv2.HoughCircles(
            gray_masked, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=30, minRadius=8, maxRadius=60
        )
        if circles is not None:
            for (cx, cy, r) in np.round(circles[0, :]).astype("int"):
                lamp = roi[cy-r:cy+r, cx-r:cx+r]
                if lamp.size == 0:
                    continue
                label, conf = predict_signal(lamp)
                if label is not None:
                    results.append((label, conf, (cx, cy, r)))  # добавили исходный цвет маски
    return results

# === Обнаружение рамки и классификация ===
def detect_frame_and_classify(image_path, save_path="result.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Image not found: {image_path}")
        return

    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)

        # эвристика: светофор вытянутый
        if 1.5 < aspect_ratio < 6 and w > 30 and h > 60:
            roi = img[y:y+h, x:x+w]

            signals = detect_signal_lights(roi)
            if not signals:
                continue

            # выбираем сигнал с max уверенностью
            label, confidence, (cx, cy, r) = max(signals, key=lambda s: s[1])
            detected = True

            # рисуем рамку светофора
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # рисуем круг активной лампы
            cv2.circle(output, (x+cx, y+cy), r, (0, 255, 0), 2)

            # пишем предсказание
            cv2.putText(output, f"{label.upper()} ({confidence*100:.1f}%)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

    if not detected:
        cv2.putText(output, "Traffic light not found", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(save_path, output)
    print(f"[INFO] Result saved in {save_path}")

# === Запуск ===
if __name__ == "__main__":
    images_dir = "images"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for i, filename in enumerate(os.listdir(images_dir), start=1):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(images_dir, filename)
            save_path = os.path.join(results_dir, f"result_{i}.jpg")

            print(f"[INFO] Processing {filename}...")
            ok = detect_frame_and_classify(img_path, save_path)

            if ok:
                print(f"[OK] Saved result to {save_path}")
            else:
                print(f"[WARN] No traffic light found in {filename}")