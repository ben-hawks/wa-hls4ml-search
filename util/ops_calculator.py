import json
import math

def calculate_dense_ops(layer_config):
    in_features = layer_config["input_shape"][1]
    out_features = layer_config["output_shape"][1]
    return out_features * in_features

def calculate_conv2d_ops(layer_config):
    input_shape = layer_config["input_shape"]
    kernel_size = layer_config["kernel_size"]

    # Calculate output shape
    output_spatial_dim = input_shape[-2] - kernel_size[0] + 1
    output_shape = (input_shape[0], layer_config["output_shape"][-1], output_spatial_dim, output_spatial_dim)

    # Calculate output_numel (total number of elements in the output tensor)
    output_numel = math.prod(output_shape[1:])

    # Calculate kernel elements
    kernel_elements = kernel_size[0] * kernel_size[1] * input_shape[-1]

    return output_numel * kernel_elements

def calculate_maxpool_ops(layer_config):
    input_shape = layer_config["input_shape"]
    output_shape = layer_config["output_shape"]

    # Derive pool size from input and output shapes
    pool_size = (
        input_shape[-2] // output_shape[-2],
        input_shape[-1] // output_shape[-1]
    )

    output_elements = output_shape[-2] * output_shape[-1] * output_shape[1]
    num_comparisons = (pool_size[0] * pool_size[1] - 1) * output_elements
    return num_comparisons

def calculate_avgpool_ops(layer_config):
    input_shape = layer_config["input_shape"]
    output_shape = layer_config["output_shape"]

    # Derive pool size from input and output shapes
    pool_size = (
        input_shape[-2] // output_shape[-2],
        input_shape[-1] // output_shape[-1]
    )

    output_elements = output_shape[-2] * output_shape[-1] * output_shape[1]
    sum_ops = (pool_size[0] * pool_size[1] - 1) * output_elements
    div_ops = output_elements
    return sum_ops + div_ops

def calculate_layer_ops(layer_config):
    class_name = layer_config.get("class_name", "")
    if class_name == "QDense":
        return calculate_dense_ops(layer_config)
    elif class_name == "QConv2D":
        return calculate_conv2d_ops(layer_config)
    elif "MaxPooling" in class_name:
        return calculate_maxpool_ops(layer_config)
    elif "AveragePooling" in class_name:
        return calculate_avgpool_ops(layer_config)
    return 0  # Default for unsupported layers

def calculate_total_ops(model_config):
    total_ops = 0
    for current_index, layer in enumerate(model_config):
        layer_ops = calculate_layer_ops(layer)
        total_ops += layer_ops
        #print(f"Layer {current_index}: {layer['class_name']} - Operations: {layer_ops}")
    return total_ops

# Example usage
if __name__ == "__main__":
    with open("example_conv.json", "r") as f:
        example_json = json.load(f)

    model_config = example_json["model_config"]

    total_ops = calculate_total_ops(model_config)
    print(f"Total Operations: {total_ops}")