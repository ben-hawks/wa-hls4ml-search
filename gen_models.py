# only supports basic layers for now
from qkeras import QDense, QConv2D, QAveragePooling2D, QActivation
from keras.layers import Input, Activation, BatchNormalization, Flatten, Layer, Dropout
from keras.models import Model

import random
import numpy as np

# rounds everything to base2
clip_base_2 = lambda x: 2**round(np.log2(x))

class ModelGenerator:

    dense_next_layers = [QDense]
    conv_next_layers = [QConv2D, QAveragePooling2D, Flatten, BatchNormalization]
    start_layers = [QDense]

    activations = [None, "relu", "tanh", "sigmoid", "softmax"]

    def config_layer(self, layer_type: Layer, bounds: dict):
        """
        Returns hyper parameters for layer initialization
        """

        if layer_type in self.dense_next_layers:
            # provides defaults if not given by call
            bounds['activation_rate'] = 0.5 if 'activation_rate' not in bounds else bounds['activation_rate']
            bounds['dropout_chance'] = .5 if 'dropout_chance' not in bounds else bounds['dropout_chance']
            bounds['dropout_rate'] = random.uniform(.1, .5) if 'dropout_rate' not in bounds else bounds['dropout_rate']
            bounds['bias_rate'] = 0.5 if 'bias_rate' not in bounds else bounds['bias_rate']

            layer_size = clip_base_2(random.randint(bounds['dense_lb'], bounds['dense_ub']))
            activation = random.choice(self.activations)
            dropout = random.random() < bounds['dropout_chance']
            use_bias = random.random() < bounds['bias_rate']
            
            hyper_params = {'size': layer_size, 'activation': activation, 'use_bias': use_bias,
                            'dropout': dropout, 'dropout_rate': bounds['dropout_rate']
                            }
        else:
            raise NotImplemented
        
        return hyper_params

    def next_layer(self, last_layer: Layer, dense_params: dict, conv_params: dict):
        """
        Takes in the current layer and will return back the next one based on what we have
        This will filter only layers that are eligible (its a form of verification)
        """

        if 'dense' in last_layer.name:
            # chooses a random layer and generates config for it. Treats layer + subsequent blocks as a unit
            layer_type = random.choice(self.dense_next_layers)
            hyper_params = self.config_layer(layer_type, dense_params)

            layer_choice = [layer_type(hyper_params['size'], 
                activation=hyper_params['activation'],
                use_bias=hyper_params['use_bias'])(last_layer)]
            
            if hyper_params['dropout']:
                layer_choice.append(Dropout(hyper_params['dropout_rate'])(layer_choice[-1]))

            layer_choice[-1].name = 'dense'
        else:
            raise NotImplemented

        return layer_choice

    def gen_network(self, total_layers: int = 3, params: dict = {'dense_lb': 32, 'dense_ub': 1024}) -> Model:
        """
        In its current form will make a dense -> dense -> ... -> dense.
        """

        # select an init layer to determine input size
        init_layer = random.choice(self.start_layers)
        layer_units = 0

        # gen size based off start layer (right now is dense so can manipulate first selection)
        input_shape = (clip_base_2(random.randint(params['dense_lb'], params['dense_ub'])),) if init_layer == QDense else 0
        layers = [Input(shape=input_shape)]

        hyper_params = self.config_layer(init_layer, params)
        if init_layer == QDense:
            layers.append(init_layer(hyper_params['size'], 
                                     activation=hyper_params['activation'],
                                     use_bias=hyper_params['use_bias'])(layers[-1]))
            layers[-1].name = "dense"
        else:
            raise NotImplemented
        
        while layer_units < total_layers:
            # disables dropout on last layer
            if layer_units == total_layers - 1:
                params['dropout_rate'] = 0
            layers.extend(self.next_layer(layers[-1], params, None))
            layer_units += 1

        # compiles the model
        model = Model(inputs=layers[0], outputs=layers[-1])
        model.build(input_shape)
        
        return model
        

if __name__ == '__main__':
    mg = ModelGenerator()
    model = mg.gen_network()

    print(model.summary())