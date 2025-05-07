import math
from bisect import bisect_left
from collections import OrderedDict

import numpy as np
import tensorflow as tf


# https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/backends/fpga/fpga_backend.py#L241
def _validate_reuse_factor(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
    _assert = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
    _assert = _assert and (((rf % n_in) == 0) or (rf < n_in))
    _assert = _assert and (((n_in * n_out) % rf) == 0)

    return _assert


# https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/backends/fpga/fpga_backend.py#L256
def get_closest_reuse_factor(n_in, n_out, chosen_rf):
    """
    Returns closest value to chosen_rf.
    If two numbers are equally close, return the smallest number.
    """

    max_rf = n_in * n_out
    valid_reuse_factors = []
    for rf in range(1, max_rf + 1):
        _assert = _validate_reuse_factor(n_in, n_out, rf)
        if _assert:
            valid_reuse_factors.append(rf)
    valid_rf = sorted(valid_reuse_factors)

    pos = bisect_left(valid_rf, chosen_rf)
    if pos == 0:
        return valid_rf[0]
    if pos == len(valid_rf):
        return valid_rf[-1]
    before = valid_rf[pos - 1]
    after = valid_rf[pos]
    if after - chosen_rf < chosen_rf - before:
        return after
    else:
        return before


def config_from_keras_model(model, reuse_factor):
    """
    _summary_

    Args:
        model (_type_): _description_
        reuse_factor (_type_): _description_

    Returns:
        _type_: _description_
    """

    if hasattr(model, "_build_input_shape") and model._build_input_shape is not None:  # Keras 2
        model_input_shape = model._build_input_shape
    elif hasattr(model, "_build_shapes_dict") and model._build_shapes_dict is not None:  # Keras 3
        model_input_shape = list(model._build_shapes_dict.values())[0]
    else:
        raise AttributeError(
            "Could not get model input shape. Make sure model.build() was called previously."
        )

    dummy_input = tf.random.uniform((1, *model_input_shape[1:]))
    _ = model(dummy_input)

    # print(model.get_config())

    layers_data = []
    for layer in model.layers:
        nested_activation = False

        class_name = layer.__class__.__name__
        layer_config = layer.get_config()
        layer_weights = layer.get_weights()

        layer_dict = {}
        layer_dict["class_name"] = class_name

        if hasattr(
            layer, "input_shape"
        ):  # Keras 2, Sequential and Functional APIs (not including subclassing)
            input_shape = layer.input_shape
        elif hasattr(layer, "_build_input_shape"):  # Keras 2, Subclassed from keras.Model
            input_shape = layer._build_input_shape

        elif hasattr(layer, "batch_shape"):  # Keras 3, InputLayer
            input_shape = layer.batch_shape
        elif hasattr(
            layer, "_build_shapes_dict"
        ):  # Keras 3, other layers (best way we found, not documented)
            input_shape = list(layer._build_shapes_dict.values())[0]

        else:
            raise AttributeError(
                f"Could not get the input shape for layer {layer.name}. Make sure model.build() was called previously."
            )

        input_shape = tuple(input_shape)
        layer_dict["input_shape"] = input_shape

        if class_name == "InputLayer":  # Same input and output shape for InputLayer
            output_shape = input_shape
        elif hasattr(layer, "output_shape"):  # Keras 2
            output_shape = layer.output_shape
        else:  # Keras 3, layers other than InputLayer
            output_shape = layer.compute_output_shape(input_shape)

        output_shape = tuple(output_shape)
        layer_dict["output_shape"] = tuple(output_shape)

        # Tracking inbound layers can be useful for add/concatenate layers
        inbound_nodes = layer.inbound_nodes
        inbound_layers = []
        for node in inbound_nodes:
            if not isinstance(node.inbound_layers, (list, tuple)):
                inbound_layers.append(node.inbound_layers)
            else:
                inbound_layers += node.inbound_layers
        layer_dict["inbound_layers"] = [layer.name for layer in inbound_layers]

        parameter_count = 0
        for weight_group in layer_weights:
            parameter_count += np.size(weight_group)

        parameter_count = int(parameter_count)
        layer_dict["parameters"] = parameter_count

        trainable_parameter_count = 0
        for var_group in layer.trainable_variables:
            trainable_parameter_count += np.size(var_group)

        trainable_parameter_count = int(trainable_parameter_count)
        layer_dict["trainable_parameters"] = trainable_parameter_count

        if class_name in ["Dense", "QDense"]:
            layer_dict["neurons"] = int(layer_config["units"])
            layer_dict["use_bias"] = layer_config["use_bias"]

        elif class_name in ["Conv1D", "Conv2D", "QConv1D", "QConv2D"]:
            layer_dict["channels"] = int(input_shape[-1])
            layer_dict["filters"] = int(layer_config["filters"])
            layer_dict["kernel_size"] = tuple([int(x) for x in layer_config["kernel_size"]])
            layer_dict["strides"] = tuple([int(x) for x in layer_config["strides"]])
            layer_dict["padding"] = layer_config["padding"]
            layer_dict["use_bias"] = layer_config["use_bias"]

        elif class_name == "Dropout":
            layer_dict["dropout_rate"] = layer_config["rate"]

        if "activation" in layer_config and layer_config["activation"] != "linear":
            if class_name in ["Activation", "QActivation"]:
                layer_dict["activation"] = layer_config["activation"]
            else:
                nested_activation = True

        dtype = layer_config["dtype"]
        if isinstance(dtype, dict):
            dtype = dtype["config"]["name"]
        layer_dict["dtype"] = dtype

        layer_dict["reuse_factor"] = reuse_factor
        if class_name in ["Dense", "QDense"]:
            n_in = np.prod([x for x in input_shape if x is not None])
            n_out = np.prod([x for x in output_shape if x is not None])
            layer_dict["reuse_factor"] = get_closest_reuse_factor(n_in, n_out, reuse_factor)
        elif class_name in ["Conv1D", "Conv2D", "QConv1D", "QConv2D"]:
            n_in = layer_dict["channels"] * np.prod(layer_dict["kernel_size"])
            n_out = layer_dict["filters"]
            layer_dict["reuse_factor"] = get_closest_reuse_factor(n_in, n_out, reuse_factor)

        layers_data.append(layer_dict)

        if (
            nested_activation
        ):  # activation function wrapped in a layer other than "Activation", example: Dense(units=32, activation="relu")
            activation_dict = {}
            activation_dict["class_name"] = "Activation"

            activation_dict["input_shape"] = output_shape
            activation_dict["output_shape"] = output_shape

            activation_dict["activation"] = layer_config["activation"]

            activation_dict["parameters"] = 0
            activation_dict["trainable_parameters"] = 0

            activation_dict["dtype"] = layer_config["dtype"]
            activation_dict["reuse_factor"] = reuse_factor

            layers_data.append(activation_dict)

    return layers_data


if __name__ == "__main__":
    import os

    import keras
    import qkeras
    from qkeras import *

    conv1d_path = os.path.join(
        os.path.dirname(__file__),
        "conv1d_run_vsynth_2023-2",
        "projects",
        "conv1d_5171_rf8192",
        "keras_model.keras"
    )
    conv2d_path = os.path.join(
        os.path.dirname(__file__),
        "conv2d_run_vsynth_2023-2",
        "projects",
        "conv2d_11852_rf8192",
        "keras_model.keras"
    )

    co = { k: v for k, v in qkeras.__dict__.items() if k[0] != "_" }
    conv1d_model = keras.models.load_model(conv1d_path, custom_objects=co)
    conv1d_model.summary()
    conv1d_config = config_from_keras_model(conv1d_model, 8192)
    print(conv1d_config)

    conv2d_model = keras.models.load_model(conv2d_path, custom_objects=co)
    conv2d_model.summary()
    conv2d_config = config_from_keras_model(conv2d_model, 8192)
    print(conv2d_config)
