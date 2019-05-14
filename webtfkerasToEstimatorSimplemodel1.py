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

SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)



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

def data_preparation(input_data_file):
    #train_data,train_label = apache_log_reader("../sdsc-http.txt",r'[SMTWF][a-z]{2} [JFMASOND][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}',"%a %b %d %H:%M:%S %Y")
    train_data,train_label = apache_log_reader(input_data_file,r'\d{2}/.../\d{4}\:\d{2}\:\d{2}\:\d{2}',"%d/%b/%Y:%H:%M:%S")
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

def data_preparation_moe(input_data_file):
    train_data, train_l, validation_data, validation_l, test_data, test_l = data_preparation(input_data_file)

    
    return train_data, train_l, validation_data, validation_l, test_data, test_l

def main1(input_data_file="../usask_access_log_50k"):

    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label = data_preparation_moe(input_data_file)
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))
    
    #print('Training laebl shape = {}'.format(len(train_label)))
    
    
    
    model = tf.keras.Sequential([
          tf.keras.layers.Dense(16,input_shape=(1,)),
          tf.keras.layers.Dense(1)
      ])
    learning_rates = [1e-2]
    adam_optimizer = Adam(lr=learning_rates[0])
    model.compile(
        loss={'dense_1': 'mean_squared_error'},
        optimizer=adam_optimizer,
        metrics=[metrics.mae]
    )

    # Print out model architecture summary
    model.summary()

    config = None
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)
    # Create an Estimator from the compiled Keras model. Note the initial model
    # state of the keras model is preserved in the created Estimator.
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model,config=config,model_dir='/tmp/multiworker/')
    
    # Treat the derived Estimator as you would with any other Estimator.
    # First, recover the input name(s) of Keras model, so we can use them as the
    # feature column name(s) of the Estimator input function:
    print("model.input_names={}".format(model.input_names))  # print out: ['input_1']
    
    # Once we have the input name(s), we can create the input function, for example,
    # for input(s) in the format of numpy ndarray:

    def train_input_fn():
        import os
        training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(tf.convert_to_tensor(train_data.values), tf.float32),
                tf.cast(tf.convert_to_tensor(train_label), tf.int32)
            )
        )
        )
        if os.environ['TF_CONFIG'] not None:
            print("sharding ....")
            training_dataset = training_dataset.shard(os.environ['NUM_WORKERS'],os.environ['WORKER_NUM'])
        training_dataset = training_dataset.batch(2000)
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