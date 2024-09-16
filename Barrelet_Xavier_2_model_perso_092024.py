import glob
import os
import shutil
import time
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


def make_model(input_shape, labels_number):
    inputs = Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        # applies a specific activation function to its input, introduce non-linearity, allowing the network to learn complex patterns
        x = layers.Activation("relu")(x)
        # SeparableConv2D layers are more parameter-efficient than standard convolutions
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        # Help with training stability and speed
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Residual connections, help in training deeper networks by allowing gradients to flow more easily.
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(labels_number, activation=None)(x)

    # From Claude, a simpler one but mine is more efficient
    # model = Sequential([
    #     Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    #     MaxPooling2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(2, 2),
    #     Conv2D(256, (3, 3), activation='relu', padding='same'),
    #     MaxPooling2D(2, 2),
    #     Flatten(),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     Dropout(0.5),
    #     Dense(num_classes, activation='softmax')
    # ])

    return Model(inputs, outputs)


if __name__ == '__main__':
    print("Starting custom model learning script.\n")

    image_size = (224, 224)
    batch_size = 32

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, validation_split=0.25, data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, validation_split=0.25, data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, data_type=None)

    with tf.device('/gpu:0'):
        model = make_model(input_shape=image_size + (3,), labels_number=120)
        # plot_model(model, show_shapes=True)

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Callbacks
        model_save_path = f"{MODELS_PATH}/custom_model_best_weights.keras"
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        callbacks_list = [checkpoint, es]

        # Fitting the model
        fitting_start_time = time.time()
        history = model.fit(dataset_train,
                            validation_data=dataset_val,
                            batch_size=batch_size,
                            epochs=5,
                            # epochs=100,  # We want early stopping to stop the training itself
                            callbacks=callbacks_list,
                            verbose=1)
        fitting_time = time.time() - fitting_start_time

        # Getting optimal epoch weights
        model.load_weights(model_save_path)

        val_loss, val_accuracy = model.evaluate(dataset_val, verbose=False)
        print(f"\nValidation Accuracy:{val_accuracy}.\n")

        test_loss, test_accuracy = model.evaluate(dataset_test, verbose=False)
        print(f"\nTest Accuracy:{test_accuracy}.\n")

        # show_history(history)
        plot_history(history, path="custom_model_results.png")
        plt.close()
