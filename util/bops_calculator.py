import json
import math


def backtrack_precision(layer_index, model_config, hls_config):
    """Backtrack through layers to find a valid precision."""
    while layer_index >= 0:
        layer = model_config[layer_index]
        class_name = layer.get("class_name", "")

        # Handle InputLayer case
        if class_name == "InputLayer":
            # print(f"Encountered InputLayer at index {layer_index}. No inbound_layers to parse.")
            return fallback_to_model(hls_config)

        inbound_layers = layer.get("inbound_layers", [])
        if not inbound_layers:
            print(f"Error: Missing or empty 'inbound_layers' for layer at index {layer_index}, layertype {class_name}.")
            return None

        layer_name = inbound_layers[0]
        if not layer_name:
            print(f"Error: Missing 'inbound_layers' for layer at index {layer_index}.")
            return None

        precision = hls_config.get("LayerName", {}).get(layer_name, {}).get("Precision", {}).get("result", "")
        if "fixed<" in precision:
            return int(precision.split("<")[1].split(",")[0])
        elif precision != "auto":
            print(f"Backtrack Error: Invalid precision format '{precision}' for layer '{layer_name}'.")
            return None

        layer_index -= 1
    print("Backtracking failed to find a valid precision.")
    fallback_to_model(hls_config)
    # Fallback to the "model" block in hls_config


def fallback_to_model(hls_config):
    model_precision = hls_config.get("Model", {}).get("Precision", {}).get("default", "")
    if "fixed<" in model_precision:
        try:
            return int(model_precision.split("<")[1].split(",")[0])
        except Exception as e:
            print(f"Error parsing model precision '{model_precision}': {e}")
            return None
    else:
        print(f"Error: Invalid model precision format '{model_precision}'.")
        return None
    print("Error: Unable to find valid precision after backtracking, including 'model' block.")
    return None

def get_bit_width_from_hls_config(hls_config, model_config, key="result", start_index=0):
    #print(f"Starting to parse model config for key '{key}' from index {start_index} to {len(model_config)}.")
    for i in range(start_index, len(model_config)):
        inbound_layers = model_config[i].get("inbound_layers", [])
        if not inbound_layers:
            print(f"Error: Missing or empty 'inbound_layers' for layer at index {i}.")
            continue
        current_layer_name = inbound_layers[0]
        precision = hls_config.get("LayerName", {}).get(current_layer_name, {}).get("Precision", {}).get(key, "")
        if precision == "auto":
            return backtrack_precision(i - 1, model_config, hls_config)
        elif "fixed<" in precision:
            return int(precision.split("<")[1].split(",")[0])

    # Handle the case for the last layer in model_config
    for layer_name in reversed(hls_config.get("LayerName", {})):
        if not layer_name.endswith("_alpha"):
            precision = hls_config["LayerName"][layer_name].get("Precision", {}).get("result", "")
            if "fixed<" in precision:
                try:
                    return int(precision.split("<")[1].split(",")[0])
                except Exception as e:
                    print(f"Error parsing precision for layer '{layer_name}': {e}")
                    return None
            elif precision == "auto":
                return backtrack_precision(len(model_config) - 1, model_config, hls_config)


    print(f"Error: No valid precision found in the configuration at index {start_index}")
    return fallback_to_model(hls_config)

def get_activation_bit_width(model_config, hls_config, start_index=0):
    try:
        #print(f"Starting to parse model config for activation bit width from index {start_index} to {len(model_config)}.")
        for i in range(start_index, len(model_config) - 1):
            next_layer = model_config[i + 1]
            inbound_layers = next_layer.get("inbound_layers", [])
            if not inbound_layers:
                print(f"Error: Missing or empty 'inbound_layers' for layer at index {i + 1}.")
                continue
            current_layer_name = inbound_layers[0]
            if current_layer_name:
                # Find the next layer with the current layer as its inbound_layer
                for layer in model_config[i + 1:]:
                    layer_inbound_layers = layer.get("inbound_layers", [])
                    if not layer_inbound_layers:
                        print(f"Error: Missing or empty 'layer_inbound_layers' for layer at index {i + 1}.")
                        continue
                    if layer_inbound_layers and layer_inbound_layers[0] == current_layer_name:# and layer.get("class_name") in ["Activation", "QActivation"]:
                        return get_bit_width_from_hls_config(hls_config, model_config, key="result", start_index=i + 1)
                    else:
                        print(f"Error: Unable to find matching inbound layer for {current_layer_name} in the next layer. Inbound layers: {layer_inbound_layers}")
                        #return get_bit_width_from_hls_config(hls_config, model_config, key="result", start_index=i + 1)

        # handle last layer
        for layer_name in reversed(hls_config.get("LayerName", {})):
            if not layer_name.endswith("_alpha"):
                precision = hls_config["LayerName"][layer_name].get("Precision", {}).get("result", "")
                if "fixed<" in precision:
                    try:
                        return int(precision.split("<")[1].split(",")[0])
                    except Exception as e:
                        print(f"Error parsing precision for layer '{layer_name}': {e}")
                        return None
                elif precision == "auto":
                    return backtrack_precision(len(model_config) - 1, model_config, hls_config)
    except ValueError as e:
        print("Unable to determine current layer name from the next layer's inbound_layer.")
        print(e)
        return None

def calculate_dense_bops(layer_config, model_config, hls_config, current_index):
    in_features = layer_config["input_shape"][1]
    out_features = layer_config["output_shape"][1]
    sparsity = layer_config.get("sparsity", 0)
    #print(layer_config)
    bit_width_w = get_bit_width_from_hls_config(hls_config, model_config, key="weight", start_index=current_index)
    bit_width_a = get_activation_bit_width(model_config, hls_config, start_index=current_index)
    #print(f"Input shape: {layer_config['input_shape']}, Output shape: {layer_config['output_shape']}")
    #print(f"Bit width W: {bit_width_w}, Bit width A: {bit_width_a}")
    return (
        out_features
        * in_features
        * ((1 - sparsity) * bit_width_a * bit_width_w + bit_width_a + bit_width_w + math.log2(in_features))
    )

def calculate_conv2d_bops(layer_config, model_config, hls_config, current_index):
    input_shape = layer_config["input_shape"]
    kernel_size = layer_config["kernel_size"]
    sparsity = layer_config.get("sparsity", 0)
    bit_width_w = get_bit_width_from_hls_config(hls_config, model_config, key="weight", start_index=current_index)
    bit_width_a = get_activation_bit_width(model_config, hls_config, start_index=current_index)

    # Calculate output shape
    output_spatial_dim = input_shape[-2] - kernel_size[0] + 1
    output_shape = (input_shape[0], layer_config["output_shape"][-1], output_spatial_dim, output_spatial_dim)

    # Calculate output_numel (total number of elements in the output tensor)
    output_numel = math.prod(output_shape[1:])

    # Calculate kernel elements
    kernel_elements = kernel_size[0] * kernel_size[1] * input_shape[-1]

    return (
        output_numel
        * kernel_elements
        * ((1 - sparsity) * bit_width_w * bit_width_a + bit_width_w + bit_width_a + math.log2(kernel_elements))
    )


def calculate_maxpool_bops(layer_config, model_config, hls_config, current_index):
    input_shape = layer_config["input_shape"]
    output_shape = layer_config["output_shape"]
    bit_width = get_bit_width_from_hls_config(hls_config, model_config, key="result", start_index=current_index)

    # Derive pool size from input and output shapes
    pool_size = (
        input_shape[-2] // output_shape[-2],
        input_shape[-1] // output_shape[-1]
    )

    output_elements = output_shape[-2] * output_shape[-1] * output_shape[1]
    num_comparisons = (pool_size[0] * pool_size[1] - 1) * output_elements
    return num_comparisons * bit_width

def calculate_avgpool_bops(layer_config, model_config, hls_config, current_index):
    input_shape = layer_config["input_shape"]
    output_shape = layer_config["output_shape"]
    bit_width = get_bit_width_from_hls_config(hls_config, model_config, key="result", start_index=current_index)

    # Derive pool size from input and output shapes
    pool_size = (
        input_shape[-2] // output_shape[-2],
        input_shape[-1] // output_shape[-1]
    )

    output_elements = output_shape[-2] * output_shape[-1] * output_shape[1]
    sum_bops = (pool_size[0] * pool_size[1] - 1) * output_elements * bit_width
    div_bops = output_elements * bit_width
    return sum_bops + div_bops

def calculate_layer_bops(layer_config, model_config, hls_config, current_index):
    class_name = layer_config.get("class_name", "")
    if class_name == "QDense":
        return calculate_dense_bops(layer_config, model_config, hls_config, current_index)
    elif class_name == "QConv2D":
        return calculate_conv2d_bops(layer_config, model_config, hls_config, current_index)
    elif "MaxPooling" in class_name:
        return calculate_maxpool_bops(layer_config, model_config, hls_config, current_index)
    elif "AveragePooling" in class_name:
        return calculate_avgpool_bops(layer_config, model_config, hls_config, current_index)
    return 0  # Default for unsupported layers

def calculate_total_bops(model_config, hls_config):
    total_bops = 0
    for current_index, layer in enumerate(model_config):
        layer_bops = calculate_layer_bops(layer, model_config, hls_config, current_index)
        total_bops += layer_bops
        #print(f"Layer {current_index}: {layer['class_name']} - BOPs: {layer_bops}")
    return total_bops

# Example usage
if __name__ == "__main__":
    with open("../dataset/fixed_rf/conv2d/conv2d_10000_rf16383_processed.json", "r") as f:
        example_json = json.load(f)

    model_config = example_json["model_config"]
    hls_config = example_json["hls_config"]

    total_bops = calculate_total_bops(model_config, hls_config)
    print(f"Total BOPs: {total_bops}")