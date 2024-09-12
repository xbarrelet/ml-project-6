import os
import shutil
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Sequential
from keras.src.applications import imagenet_utils
from keras.src.applications.vgg16 import VGG16
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.utils import image_dataset_from_directory
from plot_keras_history import show_history, plot_history

CROPPED_IMAGES_PATH = "resources/Cropped_Images"
MODELS_PATH = "models/transfer_learning"


def remove_last_generated_model():
    shutil.rmtree(MODELS_PATH, ignore_errors=True)
    os.mkdir(MODELS_PATH)


def extract_information_from_annotations(image_path):
    annotation_path = image_path.replace("Images", "Annotation").replace(".jpg", "")

    tree = ElementTree.parse(annotation_path)
    x_min = int(list(tree.iter('xmin'))[0].text)
    x_max = int(list(tree.iter('xmax'))[0].text)
    y_min = int(list(tree.iter('ymin'))[0].text)
    y_max = int(list(tree.iter('ymax'))[0].text)
    race = list(tree.iter('name'))[0].text.lower()

    return (x_min, y_min, x_max, y_max), race


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


def create_model(image_size):
    model_base = VGG16(include_top=False, weights="imagenet", input_shape=image_size + (3,))
    # Resnet, Xception
    for layer in model_base.layers:
        layer.trainable = False

    data_augmentation_layers = Sequential([
        layers.RandomFlip("horizontal", input_shape=image_size + (3,)),
        layers.RandomRotation(0.1)
    ])

    model = Sequential([
        # Data preparation
        data_augmentation_layers,
        layers.Rescaling(1. / 127.5, offset=-1),

        # Pre-trained model without the top
        model_base,

        # Convert the feature map from the encoder part of the model (without the top) to a vector
        layers.GlobalAveragePooling2D(),

        # Fully connected layers (all neurons are linked to the others).
        layers.Dense(256, activation='relu'),

        # Dropout layer to prevent overfitting. Randomly stops some neurons for each image so that the other neurons have to adapt to that, to reduce overfitting.
        # Not the latest method to do this, batch normalization for example is better. Check more online.
        layers.Dropout(0.5),

        # Output layer with the same layer as your labels. Softmax to allow the model to predict the probability of each class.
        layers.Dense(120, activation='softmax')
    ])

    # Crossentropy as we're getting closer to a correct prediction of the labels.
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    print(model.summary())

    return model


if __name__ == '__main__':
    print("Starting analysis and preprocessing script.\n")
    remove_last_generated_model()

    image_size = (224, 224)
    batch_size = 32

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, validation_split=0.25, data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, validation_split=0.25, data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, data_type=None)

    with tf.device('/gpu:0'):
        model = create_model(image_size)

        model_save_path = f"{MODELS_PATH}/model_best_weights.keras"
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        callbacks_list = [checkpoint, es]

        preprocessed_dataset_train = imagenet_utils.preprocess_input(dataset_train, mode="tf")
        history = model.fit(preprocessed_dataset_train,
                            validation_data=dataset_val,
                            batch_size=batch_size,
                            epochs=100,  # We want early stopping to stop the training itself
                            callbacks=callbacks_list,
                            verbose=1)

        # Score of last epoch
        loss, accuracy = model.evaluate(dataset_train, verbose=True)
        print(f"\nTraining Accuracy:{accuracy}.\n")
        loss, accuracy = model.evaluate(dataset_val, verbose=True)
        print(f"\nValidation Accuracy:{accuracy}.\n")

        # Score of optimal epoch
        model.load_weights(model_save_path)

        loss, accuracy = model.evaluate(dataset_val, verbose=False)
        print(f"\nValidation Accuracy:{accuracy}.\n")

        loss, accuracy = model.evaluate(dataset_test, verbose=False)
        print(f"\nTest Accuracy:{accuracy}.\n")

        show_history(history)
        plot_history(history, path="transfer_learning_history.png")
        plt.close()
