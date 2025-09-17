from flask import Flask, render_template, request, send_from_directory
import os
import cv2
from detect_traffic_light import detect_contours_tl, detect_lamps_improved, show_fixed_window

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "uploads/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result_filename = None
    signal_status = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="Файл не загружен")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="Не выбран файл")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Загружаем изображение
        img = cv2.imread(filepath)
        rois = detect_contours_tl(img, debug=False)

        signal_status = "НЕ ОБНАРУЖЕН"
        for roi in rois:
            x, y, w, h = roi
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            status = detect_lamps_improved(img, roi, debug=False)
            if status:
                signal_status = status
                cv2.putText(img, f"Signal: {status}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

        # Сохраняем результат
        result_filename = "result_" + file.filename
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, img)

    return render_template("index.html",
                       result=result_filename,
                       status=signal_status,
                       error=None)


@app.route("/results/<filename>")
def results(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
