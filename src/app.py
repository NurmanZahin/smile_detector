import logging
import tensorflow as tf
import cv2

from flask import Flask, flash, request, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
from base64 import b64encode


from .inference import predict_image
from .db import db_init
from .models import SmileImages

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('flask_app')

app = Flask(__name__)
app.secret_key = "secret key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///images.db'
db_init(app)

model_path = 'data/models/tf_mobilenetv2.h5'
loaded_model = tf.keras.models.load_model(model_path)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Getting file from post request
        file = request.files['file']
        logger.info(f"Post request received with {request.files['file']}")
        if file.filename == "":
            flash('No image uploaded')
            return redirect(request.url)

        mime_type = file.mimetype
        filename = secure_filename(file.filename)
        img = SmileImages(img=file.read(), name=filename, mime_type=mime_type)

        food_pred, confidence, face_coords = predict_image(img.img, loaded_model, face_detector)
        flash(food_pred)
        flash(round(confidence*100, 2))
        image = b64encode(img.img).decode("utf-8")
        image_uri = f"data:{mime_type};base64,{image}"
        return render_template('predict_complete.html', all_images=image_uri)
    else:
        return render_template('predict.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8000)
