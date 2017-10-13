import numpy as np
from keras.models import Sequential
from keras import initializers
from keras.layers import *
from keras.optimizers import RMSprop, SGD, Adagrad, Nadam
from keras.utils import np_utils
from keras import backend as K
import time

def relu_init(shape, name=None):
    return initializers.glorot_normal(shape)

def getModel(input_dim = 256, hidden_units = 100, hidden_layers=1, nb_classes = 256,
        lr = 0.005, out_activation='softmax', optimizer = 'Adagrad', dropout=0.0, nCaption=1):
    '''
    Create the RNN model.
    * nb_classes {integer} - The number of classes at the output layer.
    * lr {float} - The learning rate. Default is 0.01
    '''
    model = Sequential()
    if(nCaption == 1):
        model.add(Masking(mask_value=0., input_shape=(20, input_dim)))

    #Create RNN
    model.add(LSTM(output_dim = hidden_units,
                        kernel_initializer='glorot_uniform',
                        activation = 'tanh',
                        input_dim=input_dim,
                        return_sequences = True, #return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence. Not sure about this one
                        dropout_W=dropout,
                        dropout_U=dropout
                        ))

    if hidden_layers == 2:
        model.add(LSTM(output_dim = 128,
                            kernel_initializer='glorot_uniform',
                            activation = 'tanh',
                            input_dim= hidden_units,
                            return_sequences = True,
                            dropout_W=dropout,
                            dropout_U=dropout
                            ))
    # model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(nb_classes,activation=out_activation, kernel_initializer='glorot_uniform')))
    # If statements to select optimizer
    if optimizer == 'Adagrad':
        o = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    elif optimizer == 'RMSprop':
        o = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    elif optimizer == 'SDG':
        o = SGD(lr=lr, momentum = 0.9, nesterov = True)
    elif optimizer == 'Nadam':
        o = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = o,
                  metrics = ['accuracy'])

    # Store the optimizer used
    model.optimizer.__setattr__('optimizer', optimizer)

    return model
