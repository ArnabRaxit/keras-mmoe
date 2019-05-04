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
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Dense, Activation

from mmoe_tfkeras import MMoE

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
            #data.append([float(dt.timestamp())])
            data.append([float(dt.timestamp())])
            labels.append([i])
            #labels[1].append(i)
            i = i+1
    return data,labels

def data_preparation():
    #train_data,train_label = apache_log_reader("../sdsc-http.txt",r'[SMTWF][a-z]{2} [JFMASOND][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}',"%a %b %d %H:%M:%S %Y")
    train_data,train_label = apache_log_reader("../usask_access_log_50k",r'\d{2}/.../\d{4}\:\d{2}\:\d{2}\:\d{2}',"%d/%b/%Y:%H:%M:%S")
    train_data=train_data-np.amin(train_data)
    cols = ['input_1']
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

    
    return train_data, train_l, validation_data, validation_l, test_data, test_l

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
        num_tasks=1,
        name='mmoe_layers'
    )(input_layer)

    output_layers = []

    output_info = ['y0']

    # Build tower layer from MMoE layer
    #for index, task_layer in enumerate(mmoe_layers):
    tower_layer = Dense(
        units=8,
        activation='relu',
        name='tower_layer',
        kernel_initializer=VarianceScaling())(mmoe_layers)
    output_layer = Dense(
        units=1,
        name=output_info[0],
        activation='linear',
        kernel_initializer=VarianceScaling())(tower_layer) 
    output_layers.append(output_layer)

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    learning_rates = [1e-2]
    adam_optimizer = Adam(lr=learning_rates[0])
    model.compile(
        loss={'y0': 'mean_squared_error'},
        optimizer=adam_optimizer,
        metrics=[metrics.mae]
    )

    # Print out model architecture summary
    model.summary()

    
    # Create an Estimator from the compiled Keras model. Note the initial model
    # state of the keras model is preserved in the created Estimator.
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model,model_dir='/tmp/webtfkerasToEstimator')
    
    # Treat the derived Estimator as you would with any other Estimator.
    # First, recover the input name(s) of Keras model, so we can use them as the
    # feature column name(s) of the Estimator input function:
    print("model.input_names={}".format(model.input_names))  # print out: ['input_1']
    
    # Once we have the input name(s), we can create the input function, for example,
    # for input(s) in the format of numpy ndarray:

    def train_input_fn():
        training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(tf.convert_to_tensor(train_data.values), tf.float32),
                tf.cast(tf.convert_to_tensor(train_label), tf.int32)
            )
        )
        )
        training_dataset = training_dataset.batch(50000)
        #training_dataset = training_dataset.repeat(100)
        return training_dataset

    # To train, we call Estimator's train function:
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    
    # Train the model
#     model.fit(
#         x=train_data,
#         y=train_label,
#         validation_data=(validation_data, validation_label),
#         epochs=24
#     )

    predictions = estimator.predict(input_fn=train_input_fn)
    return predictions





if __name__ == '__main__':
    main()
