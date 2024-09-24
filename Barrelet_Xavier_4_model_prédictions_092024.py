import glob
import os
from pprint import pprint
from random import randint

import cv2
import keras
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.saved_model.load import load

CROPPED_IMAGES_PATH = "resources/Cropped_Images"
BEST_MODELS_PATH = "models/transfer_learning/VGG16_model.keras"


# BEST_MODELS_PATH = "models/transfer_learning/EfficientNetV2L_model.keras"


def load_image(row):
    return cv2.imread(row['image_path'], 1)


def load_images():
    images_df = DataFrame()

    all_images = list(glob.glob(f"{CROPPED_IMAGES_PATH}/*/*.jpg"))
    images_df["image_path"] = all_images

    images_df["label_name"] = images_df["image_path"].apply(lambda path: path.split("/")[-2].lower())

    labels = [f.path.split("/")[-1].lower() for f in os.scandir(CROPPED_IMAGES_PATH) if f.is_dir()]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    images_df["label"] = label_encoder.transform(images_df["label_name"])

    return images_df


def get_races_from_labels(images_df):
    races = {}

    for index in range(0, len(images_df)):
        row = dict(images_df.iloc[index])
        if row['label'] not in races:
            races[row['label']] = row['label_name']

    return races


def prepare_image_for_prediction(image_path):
    image = keras.utils.load_img(image_path, target_size=(224, 224))
    input_arr = keras.utils.img_to_array(image)
    return np.array([input_arr])  # Convert single image to a batch.


def show_image(image_path):
    image = cv2.imread(image_path, 1)
    plt.imshow(image)
    plt.show()
    plt.close()


if __name__ == '__main__':
    print("Starting inference script.\n")

    images_df = load_images()
    print("Images loaded.\n")

    races = get_races_from_labels(images_df)
    print(f"{len(races.keys())} races found.\n")

    best_model = load(BEST_MODELS_PATH)
    print("Model loaded.\n")

    for iteration in range(1, 11):
        row_id = randint(0, len(images_df))
        row = images_df.iloc[row_id]
        print(F"Predicting race of row id:{row_id}. Its race is {row['label_name']}, id:{row['label']}\n")

        show_image(row['image_path'])

        prepared_image = prepare_image_for_prediction(row['image_path'])
        predictions = best_model.serve(prepared_image)

        preds_list = list(predictions[0])
        predicted_race_id = preds_list.index(max(preds_list))
        print(f"Predicted_race id:{predicted_race_id}, name:{races[predicted_race_id]}.\n")

    print("The inference script is now finished.\n")
