# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated

import pandas as pd
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
    #train_data,train_label = apache_log_reader("./sdsc-http.txt",r'[SMTWF][a-z]{2} [JFMASOND][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}',"%a %b %d %H:%M:%S %Y")
    train_data,train_label = apache_log_reader("./usask_access_log_50k",r'\d{2}/.../\d{4}\:\d{2}\:\d{2}\:\d{2}',"%d/%b/%Y:%H:%M:%S")
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


class DataSet(object):
  """Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

  @deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
              ' from tensorflow/models.')
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    
    assert images.shape[0] == len(labels[0]), (
        'images.shape: %s labels.shape: %s' % (images.shape[0], len(labels[0])))
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images.iloc[perm0]
      self._labels = perm0
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images.iloc[start:self._num_examples]
      labels_rest_part = np.arange(start,self._num_examples)
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images.iloc[perm]
        self._labels = perm
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images.iloc[start:end]
      labels_new_part = np.arange(start,end)
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images.iloc[start:end], np.arange(start,end)


@deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
            ' from tensorflow/models.')
def read_data_sets(fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):
  
  train_data, train_labels, validation_data, validation_labels, test_data, test_labels = data_preparation_moe()

  train = DataSet(train_data, train_labels)
  validation = DataSet(validation_data, validation_labels)
  test = DataSet(test_data, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)

