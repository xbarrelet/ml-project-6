import math
import os
import shutil
import time

import keras
from keras import layers, Sequential, Input
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.optimizers import Adam, AdamW, RMSprop, SGD
from keras.src.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from pandas import DataFrame
from plot_keras_history import plot_history

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


def get_dataset(path, image_size, batch_size, validation_split=0.0, data_type=None):
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


def get_optimizer(optimizer, learning_rate):
    match optimizer:
        case "adam":
            return Adam(learning_rate=learning_rate)
        case "adamw":
            return AdamW(learning_rate=learning_rate)
        case "rmsprop":
            return RMSprop(learning_rate=learning_rate)
        case "sgd":
            return SGD(learning_rate=learning_rate)
        case "sgdn":
            return SGD(learning_rate=learning_rate, nesterov=True)
        case _:
            raise ValueError(f"Unknown optimizer:{optimizer}.")


def create_model(input_shape, labels_number, kernel_size=(3, 3), number_of_intermediate_layers=3, dropout_rate=0.2,
                 optimizer="adam", learning_rate=0.001):
    intermediate_layers = Sequential()
    for i in range(1, number_of_intermediate_layers + 1):
        intermediate_layers.add(layers.Conv2D(int(32 * math.pow(2, i)), kernel_size, activation='relu', padding='same'))
        intermediate_layers.add(layers.MaxPooling2D(2, 2))

    model = Sequential([
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

    optimizer = get_optimizer(optimizer, learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def display_results_plots(results):
    display_results_plot(results, ["fitting_time"], "fitting_time")
    display_results_plot(results, ["test_accuracy", "val_accuracy"], "accuracies", ascending=False)
    display_results_plot(results, ["test_loss", "val_loss"], "losses")


def display_results_plot(results, metrics, metrics_name, ascending=True):
    results.sort_values(metrics[0], ascending=ascending, inplace=True)

    performance_plot = (results[metrics + ["hyperparameters_name"]]
                        .plot(kind="line", x="hyperparameters_name", figsize=(15, 8), rot=0,
                              title=f"Models Sorted by {metrics_name}"))
    performance_plot.title.set_size(20)
    performance_plot.set_xticks(range(0, len(results)))
    performance_plot.set(xlabel=None)

    performance_plot.get_figure().savefig(f"{RESULTS_PATH}/{metrics_name}_plot.png", bbox_inches='tight')
    # plt.show()
    plt.close()


def get_callbacks(with_early_stopping):
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    return [checkpoint, es] if with_early_stopping else [checkpoint]


def get_results_of_model(model, dataset_train, dataset_val, dataset_test, parameters, epoch=100, batch_size=2,
                         with_early_stopping=True):
    fitting_start_time = time.time()
    history = model.fit(dataset_train,
                        validation_data=dataset_val,
                        batch_size=batch_size,
                        # epochs=2,
                        epochs=epoch,
                        callbacks=get_callbacks(with_early_stopping),
                        verbose=1)
    fitting_time = time.time() - fitting_start_time

    model.load_weights(MODEL_SAVE_PATH)

    val_loss, val_accuracy = model.evaluate(dataset_val, verbose=False)
    print(f"\nValidation Accuracy:{val_accuracy}.")

    test_loss, test_accuracy = model.evaluate(dataset_test, verbose=False)
    print(f"\nTest Accuracy:{test_accuracy}.\n")

    return {
        "hyperparameters_name": hyperparameters["name"],
        "fitting_time": fitting_time,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "history": history,
        **parameters
    }


def display_results(sorted_results, parameter_name):
    # show_history(history)
    plot_history([result['history'] for result in sorted_results], path=f"{RESULTS_PATH}/{parameter_name}_results.png")
    plt.close()



def get_best_parameter(sorted_results, parameter_name):
    best_parameter = sorted_results[0][parameter_name]
    print(f"Best parameter:{parameter_name.replace("_", " ")} found:{best_parameter}.\n")
    return best_parameter


if __name__ == '__main__':
    print("Starting custom models learning script.\n")
    remove_last_generated_models_and_results()

    image_size = (224, 224)
    batch_size = 2
    labels_number = 3

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.25, data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.25, data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, data_type=None)

    best_layers_parameters = {}

    # MODEL HYPEROPTIMIZATION
    results = []
    for hyperparameters in [
        # {"name": "1_intermediate_layers", "parameters": {"number_of_intermediate_layers": 1}}, # Too much GPU memory needed
        {"name": "2_intermediate_layers", "parameters": {"number_of_intermediate_layers": 2}},
        {"name": "3_intermediate_layers", "parameters": {"number_of_intermediate_layers": 3}},
        {"name": "4_intermediate_layers", "parameters": {"number_of_intermediate_layers": 4}},
        {"name": "5_intermediate_layers", "parameters": {"number_of_intermediate_layers": 5}}
    ]:
        print(f"\nTesting now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_model(input_shape=image_size + (3,), labels_number=labels_number,
                             **hyperparameters["parameters"])

        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    best_layers_parameters["number_of_intermediate_layers"] = get_best_parameter(sorted_results,
                                                                                 "number_of_intermediate_layers")
    display_results_plots(DataFrame(sorted_results))
    display_results(sorted_results, "number_of_intermediate_layers")

    results = []
    for hyperparameters in [
        # {"name": "kernel_layer_size_1", "parameters": {"kernel_size": (1, 1)}}, # Too much GPU memory needed
        {"name": "kernel_layer_size_2", "parameters": {"kernel_size": (2, 2)}},
        {"name": "kernel_layer_size_3", "parameters": {"kernel_size": (3, 3)}},
        {"name": "kernel_layer_size_4", "parameters": {"kernel_size": (4, 4)}},
        {"name": "kernel_layer_size_5", "parameters": {"kernel_size": (5, 5)}},
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_model(input_shape=image_size + (3,), labels_number=labels_number,
                             number_of_intermediate_layers=best_layers_parameters["number_of_intermediate_layers"],
                             **hyperparameters["parameters"])
        keras.backend.clear_session()
        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    best_layers_parameters["kernel_size"] = get_best_parameter(sorted_results, "kernel_size")
    display_results_plots(DataFrame(sorted_results))
    display_results(sorted_results, "kernel_size")

    results = []
    for hyperparameters in [
        {"name": "dropout_rate_0.1", "parameters": {"dropout_rate": 0.1}},
        {"name": "dropout_rate_0.2", "parameters": {"dropout_rate": 0.2}},
        {"name": "dropout_rate_0.3", "parameters": {"dropout_rate": 0.3}},
        {"name": "dropout_rate_0.4", "parameters": {"dropout_rate": 0.4}},
        {"name": "dropout_rate_0.5", "parameters": {"dropout_rate": 0.5}},
        {"name": "dropout_rate_0.6", "parameters": {"dropout_rate": 0.6}},
        {"name": "dropout_rate_0.7", "parameters": {"dropout_rate": 0.7}},
        {"name": "dropout_rate_0.8", "parameters": {"dropout_rate": 0.8}},
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_model(input_shape=image_size + (3,), labels_number=labels_number,
                             number_of_intermediate_layers=best_layers_parameters["number_of_intermediate_layers"],
                             kernel_size=best_layers_parameters["kernel_size"], **hyperparameters["parameters"])

        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    best_layers_parameters["dropout_rate"] = get_best_parameter(sorted_results, "dropout_rate")
    display_results(sorted_results, "dropout_rate")

    # COMPILATION HYPEROPTIMIZATION
    results = []
    for hyperparameters in [
        {"name": "rmsprop_optimizer", "parameters": {"optimizer": "rmsprop"}},
        {"name": "adam_optimizer", "parameters": {"optimizer": "adam"}},
        {"name": "adamw_optimizer", "parameters": {"optimizer": "adamw"}},
        {"name": "sgd_optimizer", "parameters": {"optimizer": "sgd"}},
        {"name": "sgd_nesterov_optimizer", "parameters": {"optimizer": "sgdn"}},
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_model(input_shape=image_size + (3,), labels_number=labels_number,
                             number_of_intermediate_layers=best_layers_parameters["number_of_intermediate_layers"],
                             kernel_size=best_layers_parameters["kernel_size"],
                             dropout_rate=best_layers_parameters["dropout_rate"],
                             **hyperparameters["parameters"])

        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    best_layers_parameters["optimizer"] = get_best_parameter(sorted_results, "optimizer")
    display_results(sorted_results, "optimizer")

    results = []
    for hyperparameters in [
        {"name": "learning_rate_0.0001", "parameters": {"learning_rate": 0.0001}},
        {"name": "learning_rate_0.0005", "parameters": {"learning_rate": 0.0005}},
        {"name": "learning_rate_0.001", "parameters": {"learning_rate": 0.001}},
        {"name": "learning_rate_0.005", "parameters": {"learning_rate": 0.005}},
        {"name": "learning_rate_0.01", "parameters": {"learning_rate": 0.01}},
        {"name": "learning_rate_0.05", "parameters": {"learning_rate": 0.05}},
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")

        model = create_model(input_shape=image_size + (3,), labels_number=labels_number,
                             number_of_intermediate_layers=best_layers_parameters["number_of_intermediate_layers"],
                             kernel_size=best_layers_parameters["kernel_size"],
                             dropout_rate=best_layers_parameters["dropout_rate"],
                             optimizer=best_layers_parameters["optimizer"],
                             **hyperparameters["parameters"])
        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    best_layers_parameters["learning_rate"] = get_best_parameter(sorted_results, "learning_rate")
    display_results(sorted_results, "learning_rate")

    # EXECUTION HYPEROPTIMIZATION
    results = []
    for hyperparameters in [
        {"name": "epoch_25", "parameters": {"epoch": 25, "with_early_stopping": False}},
        {"name": "epoch_50", "parameters": {"epoch": 50, "with_early_stopping": False}},
        {"name": "epoch_75", "parameters": {"epoch": 75, "with_early_stopping": False}},
        {"name": "epoch_100", "parameters": {"epoch": 100, "with_early_stopping": False}},
        {"name": "early_stopping", "parameters": {"epoch": 100, "with_early_stopping": True}},
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")

        model = create_model(input_shape=image_size + (3,), labels_number=labels_number,
                             number_of_intermediate_layers=best_layers_parameters["number_of_intermediate_layers"],
                             kernel_size=best_layers_parameters["kernel_size"],
                             dropout_rate=best_layers_parameters["dropout_rate"],
                             optimizer=best_layers_parameters["optimizer"],
                             learning_rate=best_layers_parameters["learning_rate"])
        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"], **hyperparameters["parameters"]))

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    best_layers_parameters["epoch"] = get_best_parameter(sorted_results, "epoch")
    best_layers_parameters["with_early_stopping"] = get_best_parameter(sorted_results, "with_early_stopping")
    display_results(sorted_results, "epoch")

    results = []
    for hyperparameters in [
        {"name": "batch_size_4", "parameters": {"batch_size": 4}},
        {"name": "batch_size_8", "parameters": {"batch_size": 8}},
        {"name": "batch_size_16", "parameters": {"batch_size": 16}},
        {"name": "batch_size_32", "parameters": {"batch_size": 32}},
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")

        model = create_model(input_shape=image_size + (3,), labels_number=labels_number,
                             number_of_intermediate_layers=best_layers_parameters["number_of_intermediate_layers"],
                             kernel_size=best_layers_parameters["kernel_size"],
                             dropout_rate=best_layers_parameters["dropout_rate"],
                             optimizer=best_layers_parameters["optimizer"],
                             learning_rate=best_layers_parameters["learning_rate"])

        results.append(get_results_of_model(model, hyperparameters["parameters"], dataset_train, dataset_val,
                                            dataset_test, epoch=best_layers_parameters["epoch"],
                                            with_early_stopping=best_layers_parameters["with_early_stopping"],
                                            **hyperparameters["parameters"]))

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    best_layers_parameters["batch_size"] = get_best_parameter(sorted_results, "batch_size")
    display_results(sorted_results, "batch_size")

    print(f"Hyperoptimization now done. Best hyperparameters found:{best_layers_parameters}.\n")
    print("Custom models learning script finished.\n")
