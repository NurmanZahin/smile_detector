import logging
import os

from flask import Flask, flash, request, jsonify, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
from base64 import b64encode
import markdown
import markdown.extensions.fenced_code

# from .inference import predict_image, create_model, init_model
from .db import db_init
from .models import FoodImages

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('flask_app')

app = Flask(__name__)
app.secret_key = "secret key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///images.db'
db_init(app)


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
        img = FoodImages(img=file.read(), name=filename, mime_type=mime_type)

        food_pred, confidence = predict_image(img.img, loaded_model)
        flash(food_pred)
        flash(round(confidence*100, 2))
        image = b64encode(img.img).decode("utf-8")
        image_uri = f"data:{mime_type};base64,{image}"
        return render_template('predict_complete.html', all_images=image_uri)
    else:
        return render_template('predict.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8000)
