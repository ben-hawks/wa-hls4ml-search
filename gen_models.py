# layers explicitly implemented by name to emphasize larger use in randomization logic
from qkeras import QDense, QConv2D, QAveragePooling2D, QActivation, QDepthwiseConv2D, QSeparableConv2D
from keras.layers import Dense, Conv2D, Flatten, Activation, Layer, Input
import keras.layers
from keras.models import Model

import random
import numpy as np

# rounds everything to base2
clip_base_2 = lambda x: 2**round(np.log2(x))

class ModelGenerator:
    # TODO: add extra q layers, quantized bits
    # TODO: add conv1d
    # TODO: Make deployable from command line (allow load in for json)
 
    dense_layers = [Dense, QDense]
    conv_layers = [Conv2D, QConv2D, QSeparableConv2D]

    start_layers = [Conv2D, QConv2D, QDense, Dense, QSeparableConv2D]
    layer_depth = 0             # counter to avoid reuse of layer names (throwaway values)

    q_on = None                 # later gets defined but here as a placeholder

    activations = ["no_activation", "relu", "tanh", "sigmoid", "softmax"]

    def config_layer(self, layer_type: Layer, bounds: dict) -> dict:
        """
        Returns hyper parameters for layer initialization as a dict

        argumnets:
        layer_type -- takes in the selection of layer so it can specify
        bounds -- specified hyperparameter limits
        """

        activation = random.choices(self.activations, weights=bounds['probs']['activations'], k=1)[0]
        bounds['bias_rate'] = 0.5 if 'bias_rate' not in bounds else bounds['bias_rate']
        use_bias = random.random() < bounds['bias_rate']
        
        if layer_type in self.dense_layers:
            # provides defaults if not given by call
            bounds['activation_rate'] = 0.5 if 'activation_rate' not in bounds else bounds['activation_rate']
            bounds['dropout_chance'] = .5 if 'dropout_chance' not in bounds else bounds['dropout_chance']
            bounds['dropout_rate'] = random.uniform(.1, .4) if 'dropout_rate' not in bounds else bounds['dropout_rate']

            layer_size = clip_base_2(random.randint(bounds['dense_lb'], bounds['dense_ub']))
            dropout = random.random() < bounds['dropout_chance']
            
            hyper_params = {'size': layer_size, 'activation': activation, 'use_bias': use_bias,
                            'dropout': dropout, 'dropout_rate': bounds['dropout_rate']
                            }
        else:
            out_filters = clip_base_2(random.randint(3, 256))
            bounds['flatten_chance'] = .5 if 'flatten_chance' not in bounds else bounds['flatten_chance']
            bounds['pooling_chance'] = .5 if 'pooling' not in bounds else bounds['pooling_chance']
            bounds['padding_chance'] = .5 if 'padding' not in bounds else bounds['padding_chance']

            flatten = random.random() < bounds['flatten_chance']
            pooling = random.random() < bounds['pooling_chance']
            padding = random.choices(["same", "valid"], weights=bounds['probs']['padding'], k=1)[0]

            # have to guarantee the range is > (1, 1)
            min_dim = min(bounds['last_layer_shape'][0], bounds['last_layer_shape'][1])
            kernel_size = random.randint(1, max(2, min_dim))
            stride = random.randint(1, 3)

            hyper_params = {'out_filters': out_filters, 'kernel': (kernel_size, kernel_size), 
                            'flatten': flatten, 'activation': activation, 'use_bias': use_bias,
                            'pooling': pooling, 'padding': padding, 'stride': (stride, stride)}
        
        return hyper_params

    def next_layer(self, last_layer: Layer, params: dict, input_layer: Layer = None) -> Layer:
        """
        Takes previous layer and configuration displays and returns back layer
        
        arguments:
        last_layer -- previous keras/qkeras layer
        params -- dictionary with specifications
        """

        if 'dense' in last_layer.name:
            # chooses a random layer and generates config for it. Treats layer + subsequent blocks as a unit
            layer_type = random.choices(self.dense_layers, weights=params['probs']['dense_layers'], k=1)[0] if input_layer == None else last_layer

            hyper_params = self.config_layer(layer_type, params)

            last_layer = last_layer if input_layer == None else input_layer
            layer_choice = [layer_type(hyper_params['size'], 
                        use_bias=hyper_params['use_bias'])(last_layer)]

            if self.q_on:
                if hyper_params['activation'] != "no_activation":
                    layer_choice.append(QActivation(activation=hyper_params['activation'], 
                                                    name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))
            else:
                if hyper_params['dropout']:
                    layer_choice.append(keras.layers.Dropout(hyper_params['dropout_rate'])(layer_choice[-1]))

                if hyper_params['activation'] != "no_activation":
                    layer_choice.append(Activation(activation=hyper_params['activation'], 
                                                    name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))

            self.layer_depth += 1
            layer_choice[-1].name = 'dense'
        elif 'conv' in last_layer.name:
            layer_type = random.choices(self.conv_layers, weights=params['probs']['conv_layers'], k=1)[0] if input_layer == None else last_layer
            params['last_layer_shape'] = last_layer.shape[1:]  if input_layer == None else params['last_layer_shape']
            hyper_params = self.config_layer(layer_type, params)

            last_layer = last_layer if input_layer == None else input_layer
            layer_choice = [layer_type(hyper_params['out_filters'], hyper_params['kernel'], strides=hyper_params['stride'],
                                    use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]

            if self.q_on:
                if hyper_params['activation'] != "no_activation":
                    layer_choice.append(QActivation(activation=hyper_params['activation'], 
                        name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))

                if hyper_params['pooling']:
                    layer_choice.append(QAveragePooling2D()(layer_choice[-1]))
            else:
                if hyper_params['activation'] != "no_activation":
                    layer_choice.append(Activation(activation=hyper_params['activation'], 
                        name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))

                if hyper_params['pooling']:
                    pooling = random.choices([keras.layers.MaxPooling2D, keras.layers.AveragePooling2D], 
                                            weights=params['probs']['pooling'], k=1)[0]
                    layer_choice.append(pooling((2, 2))(layer_choice[-1]))
            
            self.layer_depth += 1
            layer_choice[-1].name = 'conv'
            if hyper_params['flatten'] and input_layer == None:
                layer_choice.append(Flatten()(last_layer))
                layer_choice[-1].name = 'dense'

        return layer_choice

    def gen_network(self, total_layers: int = 3, 
                    add_params: dict = {}, q_chance: float = .5) -> Model:
        """
        Generates interconnected network based on defaults or extra params, returns Model

        keyword arguments:
        total_layers -- total active layers in a network (default: 3)
        add_params -- parameters to specify besides defaults for model generation (default: {})
        q_chance -- the prob that we use qkeras over keras
        """

        add_params = {k: add_params[k] for k in add_params}

        params = {'dense_lb': 32, 'dense_ub': 1024, 
                            'conv_init_size_lb': 32, 'conv_init_size_ub': 128,
                            'conv_filters_lb': 3, 'conv_filters_ub': 64, 
                            'probs': {
                                    'activations': [1 / len(self.activations) for activ_func in self.activations],
                                    'dense_layers': [], 'conv_layers': [], 'start_layers': [],
                                    'padding': [0.5, 0.5],   # border, off
                                    'pooling': [0.5, 0.5]    # max, avg
                                    }
                        }
        params.update(add_params)

        # wipe either all the qkeras or keras layers depending on what mode we're in
        self.filter_q(q_chance, params)

        init_layer = random.choices(self.start_layers, weights=params['probs']['start_layers'], k=1)[0]
        layer_units = 0

        # gen size based off start layer (right now is dense so can manipulate first selection)
        if init_layer == QDense:
            input_shape = (clip_base_2(random.randint(params['dense_lb'], params['dense_ub'])),)
        else:
            y_dim = random.randint(params['conv_init_size_lb'], params['conv_init_size_ub']) 
            x_dim = random.randint(params['conv_init_size_lb'], params['conv_init_size_ub'])
            num_filters = clip_base_2(random.randint(params['conv_filters_lb'], params['conv_filters_ub']))

            input_shape = (y_dim, x_dim, num_filters)
        layers = [Input(shape=input_shape)]

        # create the initial layer to go off of
        params['last_layer_shape'] = layers[0].shape[1:]
        init_layer.name = "dense" if init_layer in self.dense_layers else "conv"
        
        layers.extend(self.next_layer(init_layer, params, input_layer=layers[0]))
        while layer_units < total_layers:
            # disables dropout on last layer
            if layer_units == total_layers - 1:
                params['dropout_rate'] = 0
            layers.extend(self.next_layer(layers[-1], params))
            layer_units += 1

        # compiles the model
        model = Model(inputs=layers[0], outputs=layers[-1])
        model.build(input_shape)
        
        return model
    
    def reset_layers(self) -> None:
        """
        Used to return class to initial state. Useful if generating multiple networks
        """
        self.dense_layers = [Dense, QDense]
        self.conv_layers = [Conv2D, QConv2D]

        self.start_layers = [QDense, QConv2D, Dense, Conv2D]

    def filter_q(self, q_chance: float, params: dict) -> None:
        """
        Filter self.<layers> based on whether q_on is set or not

        q_chance -- the chance that we load qkeras over keras
        params -- params to dictacte flow
        """
        blacklist = []
        
        self.q_on = random.random() < q_chance
        for layer in set(self.start_layers + self.conv_layers + self.dense_layers):
            is_qkeras = layer.__module__[:6] == 'qkeras'

            if self.q_on ^ is_qkeras:
                blacklist.append(layer)
        
        self.start_layers = [layer for layer in self.start_layers if layer not in blacklist]
        self.dense_layers = [layer for layer in self.dense_layers if layer not in blacklist]
        self.conv_layers = [layer for layer in self.conv_layers if layer not in blacklist]

        # defaults if the layer was not set. Setting these is intentionally very delicate
        if params['probs']['start_layers'] == []:
            params['probs']['start_layers'] = [1 / len(self.start_layers) for layer in self.start_layers]

        if params['probs']['dense_layers'] == []:
            params['probs']['dense_layers'] = [1 / len(self.dense_layers) for layer in self.dense_layers]

        if params['probs']['conv_layers'] == []:
            params['probs']['conv_layers'] = [1 / len(self.conv_layers) for layer in self.conv_layers]

    def print_help(self):
        # defaults
        params_str = """
        params = {
            'dense_lb': 32, 'dense_ub': 1024, 
            'conv_init_size_lb': 32, 'conv_init_size_ub': 128,
            'conv_filters_lb': 3, 'conv_filters_ub': 64, 
            'probs': {
                    'activations': [1 / len(self.activations) for activ_func in self.activations],
                    'dense_layers': [uniform_dist], 'conv_layers': [uniform_dist], 'start_layers': [uniform_dist],
                    'padding': [0.5, 0.5],   # border, off
                    'pooling': [0.5, 0.5]    # max, avg
                    }
            'activation_rate': .5,                        # chances we apply activation function per layer
            'droupout_chance': .5                         # chances dropout is on
            'dropout_rate': rand[.1, .4],                 # how much to dropout if dropout on
            'flatten_chance': .5,
            'pooling_chance': .5,
            'padding_chance': .5
            """

if __name__ == '__main__':
    mg = ModelGenerator()
    for _ in range(5):
        model = mg.gen_network(add_params={'dense_ub': 64, 'conv_filters_ub': 16})
        mg.reset_layers()

        model.summary()
