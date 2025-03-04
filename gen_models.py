# layers explicitly implemented by name to emphasize larger use in randomization logic
from qkeras import QDense, QConv2D, QAveragePooling2D, QActivation, QDepthwiseConv2D, QSeparableConv2D, quantized_bits
from keras.layers import Dense, Conv2D, Flatten, Activation, Layer, Input
import keras.layers
from keras.models import Model

import random
import numpy as np

# rounds everything to base2
clip_base_2 = lambda x: 2**round(np.log2(x))

class ModelGenerator:
    # TODO: add conv1d, RNN, exotic qKeras
    # TODO: function callbacks
    # TODO (not urgent): add verbosity log file (helps with metadata for dataset generation)
    # TODO (not urgent): Make deployable from command line (allow load in for json)
 
    dense_layers = [Dense, QDense]
    conv_layers = [Conv2D, QConv2D]

    start_layers = [Conv2D, QConv2D, QDense, Dense]
    layer_depth = 0             # counter to avoid reuse of layer names (throwaway values)

    q_on = None                 # later gets defined but here as a placeholder

    activations = ["no_activation", "relu", "tanh", "sigmoid", "softmax"]

    def config_layer(self, layer_type: Layer) -> dict:
        """
        Returns hyper parameters for layer initialization as a dict

        argumnets:
        layer_type -- takes in the selection of layer so it can specify
        """

        activation = random.choices(self.activations, weights=self.params['probs']['activations'], k=1)[0]
        self.params['bias_rate'] = self.params['bias_rate']
        use_bias = random.random() < self.params['bias_rate']
        
        if layer_type in self.dense_layers:
            # provides defaults if not given by call
            self.params['activation_rate'] = self.params['activation_rate']
            self.params['dropout_chance'] = self.params['dropout_chance']
            self.params['dropout_rate'] = random.uniform(.1, .4) if 'dropout_rate' not in self.params else self.params['dropout_rate']

            layer_size = clip_base_2(random.randint(self.params['dense_lb'], self.params['dense_ub']))
            dropout = random.random() < self.params['dropout_chance']
            
            hyper_params = {'size': layer_size, 'activation': activation, 'use_bias': use_bias,
                            'dropout': dropout, 'dropout_rate': self.params['dropout_rate']
                            }
        else:
            out_filters = clip_base_2(random.randint(3, 256))
            self.params['flatten_chance'] = self.params['flatten_chance']
            self.params['pooling_chance'] = self.params['pooling_chance']
            self.params['padding_chance'] = self.params['padding_chance']

            flatten = random.random() < self.params['flatten_chance']
            pooling = random.random() < self.params['pooling_chance']
            padding = random.choices(["same", "valid"], weights=self.params['probs']['padding'], k=1)[0]

            # have to guarantee the range is > (1, 1)
            min_dim = min(self.params['last_layer_shape'][0], self.params['last_layer_shape'][1])
            kernel_size = random.randint(1, max(2, min_dim))
            stride = random.randint(self.params['conv_stride_lb'], self.params['conv_stride_ub'])

            hyper_params = {'out_filters': out_filters, 'kernel': (kernel_size, kernel_size), 
                            'flatten': flatten, 'activation': activation, 'use_bias': use_bias,
                            'pooling': pooling, 'padding': padding, 'stride': (stride, stride)}
        
        return hyper_params

    def next_layer(self, last_layer: Layer, input_layer: Layer = None) -> Layer:
        """
        Takes previous layer and configuration displays and returns back layer
        
        arguments:
        last_layer -- previous keras/qkeras layer
        params -- dictionary with specifications
        """

        if 'dense' in last_layer.name:
            # chooses a random layer and generates config for it. Treats layer + subsequent blocks as a unit
            layer_type = random.choices(self.dense_layers, weights=self.params['probs']['dense_layers'], k=1)[0] if input_layer == None else last_layer

            hyper_params = self.config_layer(layer_type)

            last_layer = last_layer if input_layer == None else input_layer
            

            if self.q_on:
                layer_choice = [layer_type(hyper_params['size'], 
                            kernel_quantizer=quantized_bits(self.params['weight_bit_width'], self.params['weight_int_width']),
                            use_bias=hyper_params['use_bias'])(last_layer)]
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(QActivation(activation=hyper_params['activation'], 
                                                    name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))
            else:
                layer_choice = [layer_type(hyper_params['size'], 
                            use_bias=hyper_params['use_bias'])(last_layer)]
                if hyper_params['dropout']:
                    layer_choice.append(keras.layers.Dropout(hyper_params['dropout_rate'])(layer_choice[-1]))

                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(Activation(activation=hyper_params['activation'], 
                                                    name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))

            self.layer_depth += 1
            layer_choice[-1].name = 'dense'
        elif 'conv' in last_layer.name:
            layer_type = random.choices(self.conv_layers, weights=self.params['probs']['conv_layers'], k=1)[0] if input_layer == None else last_layer
            self.params['last_layer_shape'] = last_layer.shape[1:]  if input_layer == None else self.params['last_layer_shape']
            hyper_params = self.config_layer(layer_type)

            last_layer = last_layer if input_layer == None else input_layer

            if self.q_on:
                layer_choice = [layer_type(hyper_params['out_filters'], hyper_params['kernel'], strides=hyper_params['stride'],
                                        kernel_quantizer=quantized_bits(self.params['weight_bit_width'], self.params['weight_int_width']),
                                        use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]

                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(QActivation(activation=hyper_params['activation'], 
                        name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))

                if hyper_params['pooling']:
                    layer_choice.append(QAveragePooling2D()(layer_choice[-1]))
            else:
                layer_choice = [layer_type(hyper_params['out_filters'], hyper_params['kernel'], strides=hyper_params['stride'],
                                        use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]

                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(Activation(activation=hyper_params['activation'], 
                        name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))

                if hyper_params['pooling']:
                    pooling = random.choices([keras.layers.MaxPooling2D, keras.layers.AveragePooling2D], 
                                            weights=self.params['probs']['pooling'], k=1)[0]
                    layer_choice.append(pooling((2, 2))(layer_choice[-1]))
            
            self.layer_depth += 1
            layer_choice[-1].name = 'conv'
            if hyper_params['flatten'] and input_layer == None:
                layer_choice.append(Flatten()(last_layer))
                layer_choice[-1].name = 'dense'

        return layer_choice

    def gen_network(self, total_layers: int = 3, 
                    add_params: dict = {}) -> Model:
        """
        Generates interconnected network based on defaults or extra params, returns Model

        keyword arguments:
        total_layers -- total active layers in a network (default: 3)
        add_params -- parameters to specify besides defaults for model generation (default: {})
        q_chance -- the prob that we use qkeras over keras
        """

        add_params = {k: add_params[k] for k in add_params}

        self.params = {'dense_lb': 32, 'dense_ub': 1024, 
                    'conv_init_size_lb': 32, 'conv_init_size_ub': 128,
                    'conv_filters_lb': 3, 'conv_filters_ub': 64, 
                    'conv_stride_lb': 1, 'conv_stride_ub': 3,
                    'q_chance': .5,
                    'activ_bit_width': 8, 'activ_int_width': 4,
                    'weight_bit_width': 6, 'weight_int_width': 3,
                    'probs': {
                            'activations': [],
                            'dense_layers': [], 'conv_layers': [], 'start_layers': [],
                            'padding': [0.5, 0.5],   # border, off
                            'pooling': [0.5, 0.5]    # max, avg
                            },
                    'activation_rate': .5,                        # chances we apply activation function per layer
                    'dropout_chance': .5,                         # chances dropout is on
                    'dropout_rate': .4,                           # how much to dropout if dropout on
                    'flatten_chance': .5,
                    'pooling_chance': .5,
                    'padding_chance': .5,
                    'bias_rate': .5
                }
        self.params.update(add_params)

        # wipe either all the qkeras or keras layers depending on what mode we're in
        self.filter_q(self.params['q_chance'], self.params)

        init_layer = random.choices(self.start_layers, weights=self.params['probs']['start_layers'], k=1)[0]
        layer_units = 0

        # gen size based off start layer (right now is dense so can manipulate first selection)
        if init_layer == QDense:
            input_shape = (clip_base_2(random.randint(self.params['dense_lb'], self.params['dense_ub'])),)
        else:
            y_dim = random.randint(self.params['conv_init_size_lb'], self.params['conv_init_size_ub']) 
            x_dim = random.randint(self.params['conv_init_size_lb'], self.params['conv_init_size_ub'])
            num_filters = clip_base_2(random.randint(self.params['conv_filters_lb'], self.params['conv_filters_ub']))

            input_shape = (y_dim, x_dim, num_filters)
        layers = [Input(shape=input_shape)]

        # create the initial layer to go off of
        self.params['last_layer_shape'] = layers[0].shape[1:]
        init_layer.name = "dense" if init_layer in self.dense_layers else "conv"
        
        layers.extend(self.next_layer(init_layer, input_layer=layers[0]))
        while layer_units < total_layers:
            # disables dropout on last layer
            if layer_units == total_layers - 1:
                self.params['dropout_rate'] = 0
            layers.extend(self.next_layer(layers[-1]))
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

        if self.q_on:
            self.activations.remove('softmax')
            self.activations = [f'quantized_{activ_func}({params["activ_bit_width"]},{params["activ_int_width"]})' for activ_func in self.activations]

        # defaults if the layer was not set. Setting these is intentionally very delicate
        pairs = {'activations': self.activations, 'start_layers': self.start_layers, 'dense_layers': self.dense_layers, 'conv_layers': self.conv_layers}
        for param_type in pairs:
            if not params['probs'][param_type]:
                params['probs'][param_type] = [1 / len(pairs[param_type]) for _ in pairs[param_type]]

if __name__ == '__main__':
    mg = ModelGenerator()
    for _ in range(1):
        model = mg.gen_network(add_params={'dense_ub': 64, 'conv_filters_ub': 16, 'q_chance': 1})
        mg.reset_layers()

        model.summary()
