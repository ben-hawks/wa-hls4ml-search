from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l1


def create_model():
    model = Sequential()
    # d_in	d_out prec rf strategy (one-hot encoded)
    model.add(Dense(64, input_shape=(6,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='leaky_relu', name='relu1'))
    #model.add(BatchNormalization())
    model.add(Dense(128, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='leaky_relu', name='relu2'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.4))
    model.add(Dense(64, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='leaky_relu', name='relu3'))
    #model.add(BatchNormalization())
    model.add(Dense(128, name='fc4', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001))) #32
    model.add(Activation(activation='leaky_relu', name='relu4'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(64, name='fc5', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001))) #32
    model.add(Activation(activation='leaky_relu', name='relu5'))
    #hls_synth_success, WorstLatency_hls, IntervalMax_hls, FF_hls, LUT_hls, BRAM_18K_hls, DSP_hls
    model.add(Dense(7, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    return model

