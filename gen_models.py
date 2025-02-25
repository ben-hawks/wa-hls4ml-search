# only supports basic layers for now
from qkeras import QDense, QConv2D, QAveragePooling2D, QActivation
from keras.layers import Dense, Conv2D, Activation, BatchNormalization, Flatten, Layer, Dropout, Input
from keras.models import Model

import random
import numpy as np

# rounds everything to base2
clip_base_2 = lambda x: 2**round(np.log2(x))

class ModelGenerator:
    # TODO: Weighting for even dist
    # TODO: quantized bits
    # TODO: support regular layers
    # TODO: add extra q layers
    # TODO: add scheduler?

    dense_next_layers_q = [QDense]
    dense_next_layers_nq = [Dense]

    conv_next_layers_q = [QConv2D]
    conv_next_layers_nq = [Conv2D]

    start_layers = [QDense, QConv2D]
    layer_depth = 0

    # whether in QKeras or regular
    q_on = True

    activations = [None, "relu", "tanh", "sigmoid", "softmax"]

    def config_layer(self, layer_type: Layer, bounds: dict) -> dict:
        """
        Returns hyper parameters for layer initialization as a dict

        argumnets:
        layer_type -- takes in the selection of layer so it can specify
        bounds -- specified hyperparameter limits
        """

        activation = random.choice(self.activations)
        bounds['bias_rate'] = 0.5 if 'bias_rate' not in bounds else bounds['bias_rate']
        use_bias = random.random() < bounds['bias_rate']
        
        if layer_type in self.dense_next_layers_q:
            # provides defaults if not given by call
            bounds['activation_rate'] = 0.5 if 'activation_rate' not in bounds else bounds['activation_rate']
            bounds['dropout_chance'] = .5 if 'dropout_chance' not in bounds else bounds['dropout_chance']
            bounds['dropout_rate'] = random.uniform(.1, .5) if 'dropout_rate' not in bounds else bounds['dropout_rate']

            layer_size = clip_base_2(random.randint(bounds['dense_lb'], bounds['dense_ub']))
            dropout = random.random() < bounds['dropout_chance']
            
            hyper_params = {'size': layer_size, 'activation': activation, 'use_bias': use_bias,
                            'dropout': dropout, 'dropout_rate': bounds['dropout_rate']
                            }
        else:
            out_filters = clip_base_2(random.randint(3, 256))
            bounds['flatten_chance'] = .5 if 'flatten_chance' not in bounds else bounds['flatten_chance']
            bounds['pooling'] = .5 if 'pooling' not in bounds else bounds['pooling']

            flatten = random.random() < bounds['flatten_chance']
            pooling = random.random() < bounds['pooling']

            # TODO: Make  extra params like stride, padding, kernel_size, etc
            kernel_size = random.randint(1, 8)

            hyper_params = {'out_filters': out_filters, 'kernel': (kernel_size, kernel_size), 
                            'flatten': flatten, 'activation': activation, 'use_bias': use_bias,
                            'pooling': pooling}
        
        return hyper_params

    def next_layer(self, last_layer: Layer, dense_params: dict, conv_params: dict) -> Layer:
        """
        Takes previous layer and configuration displays and returns back layer
        
        arguments:
        last_layer -- previous keras/qkeras layer
        dense_params -- dictionary with dense specifications
        conv_params -- dictionary with conv specifications
        """

        if 'dense' in last_layer.name:
            # chooses a random layer and generates config for it. Treats layer + subsequent blocks as a unit
            # TODO: make q vs nq choice
            layer_type = random.choice(self.dense_next_layers_q)
            hyper_params = self.config_layer(layer_type, dense_params)

            layer_choice = [layer_type(hyper_params['size'], 
                use_bias=hyper_params['use_bias'])(last_layer)]

            if self.q_on:
                if hyper_params['activation']:
                    layer_choice.append(QActivation(activation=hyper_params['activation'], 
                                                    name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))
            else:
                if hyper_params['dropout']:
                    layer_choice.append(Dropout(hyper_params['dropout_rate'])(layer_choice[-1]))

                if hyper_params['activation']:
                    layer_choice.append(Activation(activation=hyper_params['activation'], 
                                                    name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))

            self.layer_depth += 1
            layer_choice[-1].name = 'dense'
        elif 'conv' in last_layer.name:
            # TODO: make q vs nq choice
            layer_type = random.choice(self.conv_next_layers_q)
            hyper_params = self.config_layer(layer_type, conv_params)

            layer_choice = [layer_type(hyper_params['out_filters'], hyper_params['kernel'],
                                    use_bias=hyper_params['use_bias'])(last_layer)]

            if self.q_on:
                if hyper_params['activation']:
                    layer_choice.append(QActivation(activation=hyper_params['activation'], 
                        name=f"{hyper_params['activation']}_{self.layer_depth}")(layer_choice[-1]))

                if hyper_params['pooling']:
                    layer_choice.append(QAveragePooling2D()(layer_choice[-1]))
            else:
                pass
                # TODO: add all forms of pooling
                # TODO: add activations
            
            self.layer_depth += 1
            layer_choice[-1].name = 'conv'
            if hyper_params['flatten']:
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
        """

        params = {k: add_params[k] for k in add_params}

        defaults = {'dense_lb': 32, 'dense_ub': 1024, 
                            'conv_init_size_lb': 32, 'conv_init_size_ub': 128,
                            'conv_filters_lb': 3, 'conv_filters_ub': 64}
        for key in defaults:
            if key not in params:
                params[key] = defaults[key]

        # select an init layer to determine input size
        init_layer = random.choice(self.start_layers)
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
        hyper_params = self.config_layer(init_layer, params)
        if init_layer == QDense:
            layers.append(init_layer(hyper_params['size'], 
                                     activation=hyper_params['activation'],
                                     use_bias=hyper_params['use_bias'])(layers[-1]))
            layers[-1].name = "dense"
        else:
            layers.append(init_layer(hyper_params['out_filters'],
                                     hyper_params['kernel'])(layers[-1]))
            layers[-1].name = "conv"
        
        while layer_units < total_layers:
            # disables dropout on last layer
            if layer_units == total_layers - 1:
                params['dropout_rate'] = 0
            layers.extend(self.next_layer(layers[-1], params, params))
            layer_units += 1

        # compiles the model
        model = Model(inputs=layers[0], outputs=layers[-1])
        model.build(input_shape)
        
        return model
        

if __name__ == '__main__':
    mg = ModelGenerator()
    model = mg.gen_network(add_params={'dense_ub': 64})

    model.summary()