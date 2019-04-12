"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Peizhou Liao
"""

import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tf.keras import backend as K
from tf.keras import metrics
from tf.keras.optimizers import Adam
from tf.keras.initializers import VarianceScaling
from tf.keras.layers import Input, Dense
from tf.keras.models import Model
from tf.keras.models import Sequential
from tf.keras.layers import BatchNormalization,Dense, Activation

from mmoe import MMoE

import kerasplt as kp

SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.set_random_seed(SEED)
tf_session = tf.Session(graph=tf.get_default_graph())
K.set_session(tf_session)

import numpy as np
from datetime import datetime
import re
from collections import Counter

def apache_log_reader(logfile,regex,ts_format):
    data = []
    labels = []
    #labels.append([])
    #labels.append([])
    myregex = regex
    i = 0
    with open(logfile, encoding="utf8", errors='ignore') as f:
        for log in f:
            ts = re.findall(myregex,log)[0]
            dt = datetime.strptime(ts,ts_format)
            #data.append([i,dt.timestamp(),dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second])
            data.append([float(dt.timestamp())])
            labels.append(i)
            #labels[1].append(i)
            i = i+1
    return data,labels

def data_preparation():
    #train_data,train_label = apache_log_reader("../sdsc-http.txt",r'[SMTWF][a-z]{2} [JFMASOND][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}',"%a %b %d %H:%M:%S %Y")
    train_data,train_label = apache_log_reader("../usask_access_log_50k",r'\d{2}/.../\d{4}\:\d{2}\:\d{2}\:\d{2}',"%d/%b/%Y:%H:%M:%S")
    train_data=train_data-np.amin(train_data)
    cols = ['pk']
    train_data = pd.DataFrame.from_records(train_data, columns=cols)
    validation_data, validation_label = train_data,train_label
    test_data, test_label = train_data,train_label
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
def reshape(a):
    aa = []
    aa.append(a)
    aa.append(a)
    return aa        

def data_preparation_moe():
    train_data, train_l, validation_data, validation_l, test_data, test_l = data_preparation()

    
    return train_data, reshape(train_l), validation_data, reshape(validation_l), test_data, reshape(test_l)

def main1():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label = data_preparation_moe()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))
    
    #print('Training laebl shape = {}'.format(len(train_label)))
    
    
    
    # Set up the input layer
    input_layer = Input(shape=(num_features,))

    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=16,
        num_experts=8,
        num_tasks=2
    )(input_layer)

    output_layers = []

    output_info = ['y0', 'y1']

    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=1,
            name=output_info[index],
            activation='linear',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    learning_rates = [1e-4, 1e-3, 1e-2]
    adam_optimizer = Adam(lr=learning_rates[0])
    model.compile(
        loss={'y0': 'mean_squared_error', 'y1': 'mean_squared_error'},
        optimizer=adam_optimizer,
        metrics=[metrics.mae]
    )

    # Print out model architecture summary
    model.summary()

    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        epochs=100
    )
    return model


def main():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label = data_preparation()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    

    # Compile model
    model = Sequential()
    model.add(BatchNormalization(input_shape=(1,)))
    model.add(Dense(10,use_bias=True))
    model.add(Activation('relu'))
    model.add(Dense(1,use_bias=True))
    learning_rates = [1e-4, 1e-3, 1e-2]
    adam_optimizer = Adam(lr=learning_rates[0])
    model.compile(
        loss='mean_absolute_error',
        optimizer=adam_optimizer,
        metrics=[metrics.mae]
    )

    # Print out model architecture summary
    model.summary()

    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        epochs=100
        #,callbacks=[kp.plot_losses]
    )
    
    return model


if __name__ == '__main__':
    main()
