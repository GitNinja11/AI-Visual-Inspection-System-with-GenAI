import os
from flask import Flask, render_template, request
import cv2

from models.classifier import predict_defect
from genai.report_generator import generate_report

UPLOAD_FOLDER = "static/uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    report = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # OpenCV preprocessing
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, gray)

        status, confidence = predict_defect(image_path)
        report = generate_report(status, confidence)

        result = f"{status} (Confidence: {confidence:.2f})"

    return render_template("index.html", result=result, report=report, image=image_path)

if __name__ == "__main__":
    app.run(debug=True)
