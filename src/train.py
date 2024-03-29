from collections.abc import Callable
from copy import deepcopy
from os.path import exists
from typing import Tuple

import names
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from src.helpers import make_data


def replace_inputs(inputs: tf.Tensor, model: Model) -> tf.Tensor:
    """Replace the inputs of a model with a new set of input tensors.

    Args:
        inputs (tf.Tensor): Input tensor.
        model (Model): Model to use.

    Returns:
        tf.Tensor: _description_
    """
    unique_name = names.get_first_name()

    for ii in range(0, len(model.layers), 1):
        if ii < 2:
            x = inputs
            continue

        model.layers[ii]._name = model.layers[ii]._name + "-" + unique_name

        x = model.layers[ii](x)

    return x


def combine_models() -> Model:
    """Combine multiple models into a hydra configuration where there are multiple heads outputing different predictions.

    Returns:
        Model: The hydra model.
    """

    IMAGE_SIZE = 200

    model_path1 = "save/best_model_detection"
    model_path2 = "save/best_model_position"
    model_path3 = "save/best_model_angle"
    model_path4 = "save/best_model_area"

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    if exists(model_path1 + "/saved_model.pb"):
        model1 = load_model(model_path1)
    if exists(model_path2 + "/saved_model.pb"):
        model2 = load_model(model_path2)
    if exists(model_path3 + "/saved_model.pb"):
        model3 = load_model(model_path3)
    if exists(model_path4 + "/saved_model.pb"):
        model4 = load_model(model_path4)

    x1 = replace_inputs(inputs, model1)
    x2 = replace_inputs(inputs, model2)
    x3 = replace_inputs(inputs, model3)
    x4 = replace_inputs(inputs, model4)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[x1, x2, x3, x4],
    )

    model.save("save/best_combined_model.hd5")

    return model


def gen_base_model() -> Model:
    """The base model.

    Returns:
        Model: Base model.
    """

    IMAGE_SIZE = 200
    NFILTERS = 8

    CONV_PARAMS_1 = {
        "kernel_size": 3,
        "strides": 1,
        "use_bias": True,
        "padding": "same",
    }

    CONV_PARAMS_2 = {
        "kernel_size": 3,
        "strides": 2,
        "use_bias": True,
        "padding": "same",
    }

    model = Sequential()
    model.add(Reshape((IMAGE_SIZE, IMAGE_SIZE, 1), input_shape=(IMAGE_SIZE, IMAGE_SIZE)))

    for i in [2, 2, 4, 4, 6, 6, 8, 8]:
        model.add(Conv2D(NFILTERS * i, **CONV_PARAMS_1))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv2D(NFILTERS * i, **CONV_PARAMS_2))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(25))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(4))

    return model


def gen_position() -> Model:
    """Model for predicting position.  Dervied from base model.

    Returns:
        Model: Position model.
    """

    # retrieve saved model
    model_path = "save/base_model"

    if exists(model_path + "/saved_model.pb"):
        model = load_model(model_path)

    x = model.layers[-5].output
    x = Dense(100, name="d2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Activation("relu", name="relu2")(x)
    x = Dense(100, name="d3")(x)
    x = BatchNormalization(name="bn3")(x)
    x = Activation("relu", name="relu3")(x)
    predictions = Dense(2, name="d4")(x)

    model = tf.keras.Model(inputs=model.input, outputs=predictions)

    return model


def gen_area() -> Model:
    """Model for predicting area.  Dervied from base model.

    Returns:
        Model: Area model.
    """

    # retrieve saved model
    model_path = "save/base_model"

    if exists(model_path + "/saved_model.pb"):
        model = load_model(model_path)

    x = model.layers[-5].output
    x = Dense(100, name="d1")(x)
    x = Activation("relu", name="relu1")(x)
    x = Dense(2, name="d3")(x)
    predictions = Activation("tanh", name="tanh1")(x)

    model = tf.keras.Model(inputs=model.input, outputs=predictions)

    return model


def gen_detect() -> Model:
    """Model for predicting existance of a spaceship.  Dervied from base model.

    Returns:
        Model: Detection model.
    """

    # retrieve saved model
    model_path = "save/base_model"

    if exists(model_path + "/saved_model.pb"):
        model = load_model(model_path)

    x = model.layers[-5].output
    x = Dense(100, name="d1")(x)
    x = Activation("relu", name="relu1")(x)
    x = Dense(1, name="d2")(x)
    predictions = Activation("tanh", name="tanh1")(x)

    model = tf.keras.Model(inputs=model.input, outputs=predictions)

    return model


def gen_angle() -> Model:
    """Model for predicting angles.  Dervied from base model.

    Returns:
        Model: Angle model.
    """

    # retrieve saved model
    model_path = "save/base_model"

    if exists(model_path + "/saved_model.pb"):
        model = load_model(model_path)

    x = model.layers[-5].output
    x = Dense(100, name="d1")(x)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu", name="relu1")(x)
    x = Dense(2, name="d2")(x)
    x = BatchNormalization(name="bn2")(x)
    predictions = Activation("tanh", name="tanh1")(x)

    model = tf.keras.Model(inputs=model.input, outputs=predictions)

    return model


def add_angle_labels(batch_size: int, labels: np.ndarray) -> np.ndarray:
    """This process adds the angle information to the labels array using sin and cos instead of the radian values 0->2*PI.

    Args:
        batch_size (int): Batch shape.
        labels (np.ndarray): Original labels vector.

    Raises:
        ValueError: Invalid angles detected.

    Returns:
        np.ndarray: Modified labels with sin and cos.
    """

    YAW_COLUMN = 2

    # get angles
    angles = labels[:, YAW_COLUMN]
    angles = angles.reshape((batch_size, 1))

    # create sin and cos
    sin = np.sin(angles)
    cos = np.cos(angles)

    # perform angle verification
    calc_angles = np.arctan2(sin, cos)

    # Modify the angles so they are within the range of 0 -> 2*PI
    index = calc_angles < 0
    calc_angles[index] += 2 * np.pi

    # concatenate new columns
    new_labels = np.hstack((labels, sin, cos))

    # Error catching
    if np.any(np.isnan(angles) == True):
        return np.hstack((labels, sin, cos))
    if all(np.isclose(angles, calc_angles)) == False:
        raise ValueError("Invalid angles, they do not match!")

    return new_labels


def add_detection_labels(batch_size: int, labels: np.ndarray) -> np.ndarray:
    """This process converts the NaN labels into values that are trainable by the network.

    Args:
        batch_size (int): Batch shape.
        labels (np.ndarray): Original labels vector.

    Returns:
        np.ndarray: Valid predictions and labels without NaNs.
    """

    # array to check
    x = labels[:, 0]

    # detection vector
    detection_flags = np.zeros((batch_size, 1))

    # check each element and determine if an object exists
    index_a = np.isnan(x) == True
    index_b = np.isnan(x) == False
    detection_flags[index_a] = -1
    detection_flags[index_b] = 1

    # concatenate new columns
    new_labels = np.hstack((labels, detection_flags))

    return new_labels


def normalization(
    min_x: int,
    max_x: int,
    inputs: np.ndarray,
    tgt_min: float = -1.0,
    tgt_max: float = 1.0,
) -> np.ndarray:
    """Normalizes between the numbers to be between two values.

    Args:
        min_x (int): Minimum value expected in the input array.
        max_x (int): Maximum value expected in the input array.
        inputs (np.ndarray): the input array
        tgt_min (float, optional): Minimum value. Defaults to -1.0.
        tgt_max (float, optional): Maximum value. Defaults to 1.0.

    Returns:
        np.ndarray: Normalized array.
    """

    val = (tgt_max - tgt_min) * (inputs - min_x) / (max_x - min_x) + tgt_min
    val = np.clip(val, a_min=tgt_min, a_max=tgt_max)
    return val


def make_batch(
    batch_size: int = 64,
    has_spaceship: bool = True,
    noise_level: float = 0.8,
    variables: list = ["x", "y", "yaw", "width", "height", "sin", "cos"],
) -> Tuple[np.ndarray, np.ndarray]:
    """The training data is produce by this fuction.

    Args:
        batch_size (int, optional): Batch shape. Defaults to 64.
        has_spaceship (bool, optional): Flag to indicate if spaceship exists. Defaults to True.
        noise_level (float, optional): Noise level in image. Defaults to 0.8.
        variables (list, optional): Variables of interest. Defaults to ["x", "y", "yaw", "width", "height", "sin", "cos"].

    Raises:
        ValueError: Check for invalid ranges in input image.
        ValueError: Check for invalid ranges in filtered labels.
        ValueError: Check for invalid ranges in filtered labels.

    Returns:
        tuple: The image array and the filtered array.
    """

    # This data generation process has been modified to work with spaceship or no spaceship
    imgs, labels = zip(
        *[
            make_data(has_spaceship=has_spaceship, noise_level=noise_level)
            for _ in range(batch_size)
        ]
    )

    # fmt: off
    imgs        = np.stack(imgs)
    labels      = np.stack(labels)
    imgs        = 2 * imgs - 1       # normalize image
    labels      = add_angle_labels(batch_size, labels)
    labels      = add_detection_labels(batch_size, labels)
    all_names   = ["x", "y", "yaw", "width", "height", "sin", "cos", "detection"]

    # normalization of outputs
    x           = normalization(min_x=10, max_x=190,        inputs=labels[:, 0])
    y           = normalization(min_x=10, max_x=190,        inputs=labels[:, 1])
    yaw         = normalization(min_x=0,  max_x=2*np.pi,    inputs=labels[:, 2])
    width       = normalization(min_x=18, max_x=36,         inputs=labels[:, 3])
    height      = normalization(min_x=18, max_x=75,         inputs=labels[:, 4])
    sin         = normalization(min_x=-1, max_x=1,          inputs=labels[:, 5])
    cos         = normalization(min_x=-1, max_x=1,          inputs=labels[:, 6])
    detection   = normalization(min_x=-1, max_x=1,          inputs=labels[:, 7])

    # new labels
    x           = x.reshape((batch_size, 1))
    y           = y.reshape((batch_size, 1))
    yaw         = yaw.reshape((batch_size, 1))
    width       = width.reshape((batch_size, 1))
    height      = height.reshape((batch_size, 1))
    sin         = sin.reshape((batch_size, 1))
    cos         = cos.reshape((batch_size, 1))
    detection   = detection.reshape((batch_size, 1))
    all_vects   = [x, y, yaw, width, height, sin, cos, detection]
    # fmt: on

    # generate my labels
    my_labels = []
    for name, vector in zip(all_names, all_vects):
        if name in variables:
            my_labels.append(vector)

    filter_labels = np.hstack(tuple(my_labels))

    # checks
    check0 = imgs.min() < -1.0
    check1 = imgs.max() > 1.0
    check2 = filter_labels.shape[0] != batch_size
    check3 = filter_labels.shape[1] != len(variables)
    check4 = filter_labels.max() > 1.0
    check5 = filter_labels.min() < -1.0

    if check0 or check1:
        raise ValueError("Error in image range")
    if check2 or check3:
        raise ValueError("Error in shape")
    if check4 or check5:
        raise ValueError("Values are outside the normal ranges")

    # add two new columns for representing
    return imgs, filter_labels


class CustomSaverPred(keras.callbacks.Callback):
    """Custom Keras callback for saving data."""

    def on_epoch_end(self, epoch, logs={}):
        self.model.save("save/model{}.hd5".format(epoch))


def train_model(
    batch_size: int = 64,
    model_path: str = "save/",
    model_name: str = "saved_model.pb",
    steps_per_epoch: int = 250,
    epochs: int = 50,
    loss: object = keras.losses.MeanSquaredError(),
    optimizer: object = keras.optimizers.Adam(),
    variables: list = ["x", "y", "yaw", "width", "height", "sin", "cos", "detection"],
    has_spaceship: bool = True,
    base_model: Callable = gen_base_model,
):
    """Performing training on model.

    Args:
        batch_size (int, optional): Batch shape. Defaults to 64.
        model_path (str, optional): Path to model. Defaults to "save/".
        model_name (str, optional): Name of model. Defaults to "saved_model.pb".
        steps_per_epoch (int, optional): Number of training steps per epoch. Defaults to 250.
        epochs (int, optional): Number of epochs to train. Defaults to 50.
        loss (object, optional): Loss function. Defaults to keras.losses.MeanSquaredError().
        optimizer (object, optional): Optimizer to use. Defaults to keras.optimizers.Adam().
        variables (list, optional): Variables of interest. Defaults to ["x", "y", "yaw", "width", "height", "sin", "cos", "detection"].
        has_spaceship (bool, optional): Flag to indicate spaceship exists. Defaults to True.
        base_model (Callable, optional): The base model to use. Defaults to gen_base_model.
    """
    # define callbacks
    saver = CustomSaverPred()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    # retrieve saved model
    if exists(model_path + "/" + model_name):
        print("INFO: LOADING AN EXISTING MODEL")
        model = load_model(model_path)
    else:
        print("INFO: GENERATING A NEW MODEL")
        model = base_model()

    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    print(f"Learning Rate: {K.eval(model.optimizer.lr)}")
    model.fit_generator(
        iter(
            lambda: make_batch(
                batch_size=batch_size,
                has_spaceship=has_spaceship,
                noise_level=0.8,
                variables=variables,
            ),
            None,
        ),
        callbacks=[checkpoint],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )


def train_detection_model():
    """Train a detection model.  This model is only concerned with determining whether a spaceship exists in the noise."""
    BATCH_SIZE = 128
    MODEL_PATH = "save/best_model_detection"

    # optimizer settings
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    loss = keras.losses.MeanSquaredError()

    train_model(
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH,
        steps_per_epoch=100,
        epochs=50,
        loss=loss,
        optimizer=adam,
        variables=["detection"],
        has_spaceship=None,
        base_model=gen_detect,
    )


def train_area_model():
    """Train an area model.  This model is only concerned with predicting the $width$ and $height$ of the spaceship."""
    BATCH_SIZE = 64
    MODEL_PATH = "save/best_model_area"

    # optimizer settings
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    loss = keras.losses.MeanSquaredError()

    train_model(
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH,
        steps_per_epoch=100,
        epochs=50,
        loss=loss,
        optimizer=adam,
        variables=["width", "height"],
        base_model=gen_area,
    )


def train_position_model():
    """Train a position model.  This model is only concerned with predicting the $x$ and $y$ position of the spaceship."""
    BATCH_SIZE = 128
    MODEL_PATH = "save/best_model_position"

    # optimizer settings
    adam = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
    loss = keras.losses.MeanSquaredError()

    train_model(
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH,
        steps_per_epoch=100,
        epochs=100,
        loss=loss,
        optimizer=adam,
        variables=["x", "y"],
        base_model=gen_position,
    )


def train_angle_model():
    """Train an angle model.  This model is only concerned with predicting the angle of the spaceship."""
    BATCH_SIZE = 128
    MODEL_PATH = "save/best_model_angle"

    # optimizer settings
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    loss = keras.losses.MeanSquaredError()

    train_model(
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH,
        steps_per_epoch=100,
        epochs=50,
        loss=loss,
        optimizer=adam,
        variables=["sin", "cos"],
        base_model=gen_angle,
    )


def train_base_model():
    """Train a base model.  This model is the foundation model which the other models will use for fine-turning their predictions.  The base model is trained with more variables than the other derived models because it should be useful for all of the variables of interest."""
    BATCH_SIZE = 64
    MODEL_PATH = "save/base_model"

    # optimizer settings
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    loss = keras.losses.MeanSquaredError()

    train_model(
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH,
        steps_per_epoch=100,
        epochs=50,
        loss=loss,
        optimizer=adam,
        variables=["x", "y", "height", "width"],
        base_model=gen_base_model,
    )


def main():
    """Main function for training all models."""
    # train base model
    train_base_model()

    # train individual models
    train_detection_model()
    train_position_model()
    train_area_model()
    train_angle_model()

    # create composite model
    combine_models()


if __name__ == "__main__":
    main()
