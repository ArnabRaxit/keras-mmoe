#https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
import tensorflow as tf

import os
import json




#os.environ["TF_CONFIG"] = json.dumps({
#     "cluster": {
#         "worker": ["locahost:4792", "localhost:4793"],
#         "chief": ["localhost:4791"]
#    },
#    "task": {"type": "worker", "index": 1}
#})

NUM_WORKERS = 1
IP_ADDRS = ['localhost']
PORTS = [12345]

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)]
    },
    'task': {'type': 'worker', 'index': 0}
})




#import keras_model_to_estimator as kmte
#kmte.main("/tmp/kmtechkpt")

import  webtfkerasToEstimator as kmte
kmte.main1()
