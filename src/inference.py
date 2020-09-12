import numpy as np
import argparse
import tensorflow as tf
import io
import logging

from PIL import Image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import ResNet50V2


logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('inference')

FOODS = ['Chilli Crab',
         'Curry Puff',
         'Dim Sum',
         'Ice Kacang',
         'Kaya Toast',
         'Nasi Ayam',
         'Popiah',
         'Roti Prata',
         'Sambal Stingray',
         'Satay',
         'Tau Huay',
         'Wanton Noodle']


def create_model(num_classes, input_shape):
    """
    Creates the model architecture for weights to be loaded

    Parameters:
    ----------
    num_classes (int): the number of classes to be classified for tensorfood is 12
    input_shape (tuple): shape is pre-defined to be 450x450x3

    Returns:
    ----------
    model (tf.keras model): the compiled model
    """
    conv_base = ResNet50V2(weights=None, include_top=False, input_shape=input_shape)
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D(name='gap'))
    model.add(layers.Dense(num_classes, activation='softmax', name='final_fc'))

    model.layers[0].trainable = False

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-2),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def init_model(model, model_path):
    """Loads a model using keras load model using input file_path"""
    model.load_weights(model_path)
    return model


def preprocess_image(img):
    """Takes img as input and outputs the normalized array form"""
    img_array = np.array(img, dtype=np.uint8) / 255
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def load_input_img(img_data):
    """Loads an image from the input file path and resize to 450, 450"""
    if isinstance(img_data, str):
        with open(img_data, 'rb') as file:
            img_bytes = file.read()
    elif isinstance(img_data, (bytes, bytearray)):
        img_bytes = img_data
    else:
        img_bytes = img_data.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((450, 450))
    return img


def classify_img(classifier, img_arr, class_names):
    """Predicts on the input img array and returns the classification

    Parameters:
    ----------
    classifier: the model/classifier to be used
    imr_arr (numpy array): the preprocessed image in numpy array form
    class_names (list): the list of food names/categories

    Returns:
    ----------
    classification (string): the category of the image
    confidence (float): the confidence of the prediction
    """
    preds = classifier.predict(img_arr)
    pred_idx = np.argmax(classifier.predict(img_arr))
    classification = class_names[pred_idx]
    confidence = preds[0][pred_idx]
    return classification, confidence


def predict_image(img_file_path, model, class_names=FOODS):
    """
    Takes a file path to the image as input and predicts the type of food

    Parameters:
    ----------
    file_path (string): file path to the image to be predicted
    img_file_path (string): file path to the image to be predicted
    class_names (list): list of strings of each food type

    Returns:
    food_type (string): the predicted type of food based on input image
    """
    logger.info(f'Loading image {img_file_path}')
    img = load_input_img(img_file_path)
    img_array = preprocess_image(img)
    food_type, conf = classify_img(model, img_array, class_names)
    logger.info(f'Prediction Complete')
    return food_type, conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file_path', type=str)

    args = parser.parse_args()
    image_path = args.img_file_path
    predict_image(image_path)
