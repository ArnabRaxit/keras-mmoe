import tensorflow as tf

import os
import json
NUM_WORKERS = 3
IP_ADDRS = ['localhost','localhost','localhost']
PORTS = [12345,12346,12347]

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)]
    },
    'task': {'type': 'worker', 'index': 0}
})




#import keras_model_to_estimator as kmte
#kmte.main("/tmp/kmtechkpt")

import  webtfkerasToEstimatorSimplemodel1 as  kmte
kmte.main1()
