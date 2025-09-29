import os
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename

import logging
logging.basicConfig(level=logging.INFO, filename="logs/app.log", filemode="a",
                    format="%(asctime)s %(levelname)s: %(message)s")

from detect_traffic_light import predict_image, draw_prediction_on_image, save_result_db, model_exists

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = os.path.join(UPLOAD_FOLDER, "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-with-a-random-secret"  # для flash-сообщений


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if not model_exists():
            flash("Модель не найдена. Сначала обучите модель (train.py).")
            return render_template("index.html", result=None, status=None)

        if "file" not in request.files:
            flash("Файл не найден в запросе.")
            return render_template("index.html", result=None, status=None)

        file = request.files["file"]
        if file.filename == "":
            flash("Файл не выбран.")
            return render_template("index.html", result=None, status=None)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            try:
                pred_class, conf, probs = predict_image(save_path)
            except Exception as e:
                logging.exception("Prediction error")
                flash(f"Ошибка при прогнозировании: {e}")
                return render_template("index.html", result=None, status=None)

            out_name = "result_" + filename
            out_path = os.path.join(RESULT_FOLDER, out_name)
            draw_prediction_on_image(save_path, out_path, pred_class, conf)

            # Save to DB
            save_result_db(filename, pred_class, conf, success=1)

            return render_template("index.html", result=out_name, status=f"{pred_class} ({conf*100:.1f}%)")

        else:
            flash("Неподдерживаемый формат.")
    return render_template("index.html", result=None, status=None)


@app.route("/results/<filename>")
def results(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
