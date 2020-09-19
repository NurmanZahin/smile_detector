import numpy as np
import argparse
import cv2
import tensorflow as tf
import io
import logging
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('inference')
IMAGE_SIZE = (224, 224)


def face_detect(face_detector, img):
    """Takes a image path as input and a face detector and outputs the grayed image and the coordinates of the face detected"""
    test_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    grayed_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_detector.detectMultiScale(grayed_img, 1.1, 5)
    return grayed_img, face_coordinates


def extract_face(img, face_coordinates):
    """Takes an image and face coordinates as input and crops out the face and does the mobilenet preprocessing required"""
    if len(face_coordinates)==0:
        return 'No face detected'
    for (x, y, w, h) in face_coordinates:
        extracted_face = cv2.resize(img[y:y+h, x:x+w], (224, 224))
        extracted_face = cv2.cvtColor(extracted_face, cv2.COLOR_GRAY2RGB)
        extracted_face = preprocess_input(extracted_face)

    return extracted_face


def load_input_img(img_data):
    """Loads an image from the input file path and resize to 450, 450"""
    if isinstance(img_data, str):
        with open(img_data, 'rb') as file:
            img_bytes = file.read()
    elif isinstance(img_data, (bytes, bytearray)):
        img_bytes = img_data
    else:
        img_bytes = img_data.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return img


def classify_img(classifier, img_arr):
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
    labels = ('not smiling', 'smiling')
    prediction = classifier.predict(img_arr)[0]
    pred_label = np.argmax(prediction)
    confidence = prediction[pred_label]
    return labels[pred_label], confidence


def predict_image(img_file_path, model, face_detector):
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
    loaded_img = load_input_img(img_file_path)
    grayed_img, face_coord = face_detect(face_detector, loaded_img)

    if len(face_coord) == 0:
        return 'Face not detected'

    else:
        face_extract = extract_face(grayed_img, face_coord)
        img_array = np.expand_dims(face_extract, axis=0)
        smile_type, conf = classify_img(model, img_array)
    logger.info(f'Prediction Complete')
    return smile_type, conf, face_coord


def visualize_classifier(img_path, smile_classifier, face_detector):
    font_scale = 1.2
    font = cv2.FONT_HERSHEY_PLAIN

    smile_class, conf, face_coor = predict_image(img_path, smile_classifier, face_detector)
    img = cv2.imread(img_path)

    msg = f'{smile_class}, {conf:.3f}'
    t_size = cv2.getTextSize(smile_class, 0, fontScale=font_scale, thickness=1)[0]
    for (x, y, w, h) in face_coor:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + t_size[0], y-t_size[1]-3), (0,0,0), -1)  # filled
        cv2.putText(img, msg, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file_path', type=str)
    parser.add_argument('--model_path', type=str, default='data/models/tf_mobilenetv2.h5')
    parser.add_argument('--face_path', type=str, default='haarcascade_frontalface_default.xml')

    args = parser.parse_args()
    image_path = args.img_file_path
    model_path = args.model_path
    face_det_path = args.face_path
    test_model = tf.keras.models.load_model(model_path)
    face_detector = cv2.CascadeClassifier(face_det_path)
    # predict_image(image_path, test_model, face_detector)
    visualize_classifier(image_path, test_model, face_detector)
