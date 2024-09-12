import glob
import os
import shutil
from xml.etree import ElementTree

import tensorflow as tf
from PIL import Image
from keras import layers, Sequential
from keras.src.applications.vgg16 import VGG16
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.utils import image_dataset_from_directory
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from plot_keras_history import show_history, plot_history
import matplotlib.pyplot as plt

IMAGES_PATH = "resources/Images"
CROPPED_IMAGES_PATH = "resources/Cropped_Images"
MODELS_PATH = "models"

data_augmentation_layers = Sequential([
    layers.RandomFlip("horizontal", input_shape=(200, 200, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def extract_information_from_annotations(image_path):
    annotation_path = image_path.replace("Images", "Annotation").replace(".jpg", "")

    tree = ElementTree.parse(annotation_path)
    x_min = int(list(tree.iter('xmin'))[0].text)
    x_max = int(list(tree.iter('xmax'))[0].text)
    y_min = int(list(tree.iter('ymin'))[0].text)
    y_max = int(list(tree.iter('ymax'))[0].text)
    race = list(tree.iter('name'))[0].text.lower()

    return (x_min, y_min, x_max, y_max), race


def extract_cropped_images():
    shutil.rmtree(CROPPED_IMAGES_PATH, ignore_errors=True)
    os.makedirs(CROPPED_IMAGES_PATH, exist_ok=True)

    for image_path in glob.glob(f'{IMAGES_PATH}/*/*.jpg'):
        dimensions, race = extract_information_from_annotations(image_path)

        original_image = Image.open(image_path)
        cropped_image = original_image.crop(dimensions)

        cropped_image_path = f"{CROPPED_IMAGES_PATH}/{race}/" + image_path.split("/")[-1]
        os.makedirs(cropped_image_path.replace(image_path.split("/")[-1], ""), exist_ok=True)

        try:
            cropped_image.save(cropped_image_path)
        except OSError as e:
            if e.args == ('cannot write mode RGBA as JPEG',):
                print(f"Converting {cropped_image_path} to RGB.")
                cropped_image = cropped_image.convert('RGB')
                cropped_image.save(cropped_image_path)

    print(f"All cropped images have been extracted and saved under:{CROPPED_IMAGES_PATH}.\n")


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


def print_images_dimensions(images_df):
    dimensions = []

    for image_path in images_df[['image_path']].values:
        image = Image.open(image_path[0])
        width, height = image.size

        dimensions.append({"width": width, "height": height})

    dimensions_df = DataFrame(dimensions)
    print(dimensions_df.describe())


def get_dataset(path, validation_split=0.0, data_type=None):
    return image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        batch_size=batch_size,
        image_size=(200, 200),
        seed=42,
        validation_split=validation_split,
        subset=data_type
    )


def create_model():
    model_base = VGG16(include_top=False, weights="imagenet", input_shape=(200, 200, 3))
    for layer in model_base.layers:
        layer.trainable = False

    model = Sequential([
        data_augmentation_layers,
        layers.Rescaling(1. / 127.5, offset=-1),
        model_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(120, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    print(model.summary())

    return model


if __name__ == '__main__':
    print("Starting analysis and preprocessing script.\n")

    # extract_cropped_images()

    # images_df = load_images()
    # print(f"{len(images_df)} images have been loaded with {len(images_df['label_name'].unique())} different labels.\n")

    # print("Number of images per label:")
    # print(images_df.groupby("label_name").count())

    # print_images_dimensions(images_df)


    image_size = (200, 200)
    batch_size = 32

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, validation_split=0.25, data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, validation_split=0.25, data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, data_type=None)

    # Création du callback
    model_save_path = f"{MODELS_PATH}/model_best_weights.keras"
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [checkpoint, es]


    with tf.device('/gpu:0'):
        model = create_model()

        history = model.fit(dataset_train,
                            validation_data=dataset_val,
                            batch_size=batch_size,
                            epochs=50,
                            callbacks=callbacks_list,
                            verbose=1)

        # Score of last epoch
        loss, accuracy = model.evaluate(dataset_train, verbose=True)
        print("Training Accuracy   : {:.4f}".format(accuracy))
        print()
        loss, accuracy = model.evaluate(dataset_val, verbose=True)
        print("Validation Accuracy :  {:.4f}".format(accuracy))

        # Score of optimal epoch
        model.load_weights(model_save_path)

        loss, accuracy = model.evaluate(dataset_val, verbose=False)
        print("Validation Accuracy :  {:.4f}".format(accuracy))

        loss, accuracy = model.evaluate(dataset_test, verbose=False)
        print("Test Accuracy       :  {:.4f}".format(accuracy))

        show_history(history)
        plot_history(history, path="transfer_learning_history.png")
        plt.close()
