def config_layer(layer_selection):
    """
    Returns parameters for layer initialization
    """
    # conv, dense, (strides, padding, etc)
    
    # return hyperparams for layer
    pass

def next_layer(last_layer):
    """
    Takes in the current layer and will return back the next one based on what we have
    This will filter only layers that are eligible (its a form of verification)
    """
    # (eventually will filter the possible ones with the overall desired config)

    # with conv -> conv, flatten, batchNorm, pool
    # with dense -> dense, activation, dropout

    # config_layer() to actually generate hyperparameters

    # return layer connected to last
    pass

def gen_network(valid_layers):
    """
    In its current form will make a CNN, pool, CNN, batch norm, flatten, dense.
    """

    # init with Input layer
    
    # call next_layers -> should return all CNN but choose based on CNN/dense
    # choose layer (conv)
    
    # connect to current model

    # while loop
    #   next_layers() -> returns pool, batch_norm, flatten or dense depending on what we have
    #   connect to current model and logic to determine if its done

    # return QKeras model
    pass

if __name__ == '__main__':
    pass