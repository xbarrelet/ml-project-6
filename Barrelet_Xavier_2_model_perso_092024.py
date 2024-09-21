import glob
import math
import os
import shutil
import time
from pprint import pprint
from xml.etree import ElementTree

import keras
import tensorflow as tf
from PIL import Image
from keras import layers, Sequential, Input, Model
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.optimizers import Adam
from keras.src.utils import image_dataset_from_directory, plot_model
from matplotlib import pyplot as plt
from pandas import DataFrame
from plot_keras_history import show_history, plot_history
from sklearn.preprocessing import LabelEncoder

IMAGES_PATH = "resources/Images"
CROPPED_IMAGES_PATH = "resources/Cropped_Images2"
MODELS_PATH = "models/custom_model"
MODEL_SAVE_PATH = f"{MODELS_PATH}/custom_model.keras"
RESULTS_PATH = "results/custom_model"



def remove_last_generated_models_and_results():
    shutil.rmtree(MODELS_PATH, ignore_errors=True)
    os.makedirs(MODELS_PATH)

    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    os.makedirs(RESULTS_PATH)


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


def get_dataset(path, image_size, validation_split=0.0, data_type=None):
    return image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        batch_size=batch_size,
        image_size=image_size,
        seed=42,
        validation_split=validation_split,
        subset=data_type
    )


def create_model(input_shape, labels_number, kernel_size=(3, 3), number_of_intermediate_layers=3, dropout_rate=0.2):

    intermediate_layers = Sequential()
    for i in range(1, number_of_intermediate_layers + 1):
        intermediate_layers.add(layers.Conv2D(int(32 * math.pow(2, i)), kernel_size, activation='relu', padding='same'))
        intermediate_layers.add(layers.MaxPooling2D(2, 2))

    return Sequential([
        Input(shape=input_shape),

        # Data augmentation layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

        intermediate_layers,

        layers.Flatten(),
        layers.Dropout(dropout_rate),

        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),

        layers.Dense(labels_number, activation='softmax')
    ])


def display_results_plots(results):
    display_results_plot(results, ["fitting_time"], "fitting_time")
    display_results_plot(results, ["test_accuracy", "val_accuracy"], "accuracies", ascending=False)
    display_results_plot(results, ["test_loss", "val_loss"], "losses")


def display_results_plot(results, metrics, metrics_name, ascending=True):
    results.sort_values(metrics[0], ascending=ascending, inplace=True)

    performance_plot = (results[metrics + ["model_name"]]
                        .plot(kind="line", x="model_name", figsize=(15, 8), rot=0,
                              title=f"Models Sorted by {metrics_name}"))
    performance_plot.title.set_size(20)
    performance_plot.set_xticks(range(0, len(results)))
    performance_plot.set(xlabel=None)

    performance_plot.get_figure().savefig(f"{RESULTS_PATH}/{metrics_name}_plot.png", bbox_inches='tight')
    # plt.show()
    plt.close()


def get_callbacks():
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    return [checkpoint, es]


if __name__ == '__main__':
    print("Starting custom models learning script.\n")
    remove_last_generated_models_and_results()

    image_size = (224, 224)
    batch_size = 32
    labels_number = 4

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, validation_split=0.25, data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, validation_split=0.25, data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, data_type=None)

    # Layers hyperoptimization
    results = []
    for hyperparameters in [
        {"name": "one_intermediate_layer", "parameters":{"number_of_intermediate_layers": 1}},
        {"name": "two_intermediate_layers", "parameters":{"number_of_intermediate_layers": 2}},
        {"name": "three_intermediate_layers", "parameters":{"number_of_intermediate_layers": 3}},

        {"name": "kernel_layer_size_one", "parameters":{"kernel_size": (1,1)}},
        {"name": "kernel_layer_size_two", "parameters":{"kernel_size": (2,2)}},
        {"name": "kernel_layer_size_three", "parameters":{"kernel_size": (3,3)}},

        {"name": "dropout_rate_0.1", "parameters":{"dropout_rate": 0.1}},
        {"name": "dropout_rate_0.2", "parameters":{"dropout_rate": 0.2}},
        {"name": "dropout_rate_0.5", "parameters":{"dropout_rate": 0.5}},
    ]:
        with tf.device('/gpu:0'):
            model = create_model(input_shape=image_size + (3,), labels_number=labels_number,
                                 **hyperparameters["parameters"])

            # # TODO: Use something else than Adam, le learning rate as well
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            fitting_start_time = time.time()
            # TODO: Avec/sans early stopping et 25-50-75-100 epochs.
            # Essaie aussi avec batch size 16-32-64
            history = model.fit(dataset_train,
                                validation_data=dataset_val,
                                batch_size=batch_size,
                                epochs=2,
                                # epochs=100,  # We want early stopping to stop the training itself
                                callbacks=get_callbacks(),
                                verbose=1)
            fitting_time = time.time() - fitting_start_time

            # Getting optimal epoch weights
            model.load_weights(MODEL_SAVE_PATH)

            val_loss, val_accuracy = model.evaluate(dataset_val, verbose=False)
            print(f"\nValidation Accuracy:{val_accuracy}.\n")

            test_loss, test_accuracy = model.evaluate(dataset_test, verbose=False)
            print(f"\nTest Accuracy:{test_accuracy}.\n")

            results.append({
                "hyperparameters_name": hyperparameters["name"],
                "fitting_time": fitting_time,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss
            })

            # show_history(history)
            plot_history(history, path=f"{hyperparameters["name"]}_results.png")
            plt.close()

        display_results_plots(DataFrame(results))

    print("Custom models learning script finished.\n")
