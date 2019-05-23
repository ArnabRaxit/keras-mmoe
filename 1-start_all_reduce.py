import tensorflow as tf

import os
import json
import sys

NUM_WORKERS = len(sys.argv)-2
WORKER_NUM = sys.argv[1]
IP_ADDRS_PORTS = sys.argv[2:]  
#['localhost','localhost','localhost']
#PORTS = [12345,12346,12347]
os.environ['NUM_WORKERS'] = str(NUM_WORKERS)
os.environ['WORKER_NUM'] = WORKER_NUM

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['%s:%d' % (IP_ADDRS_PORTS[w].split(":")[0], int(IP_ADDRS_PORTS[w].split(":")[1])) for w in range(NUM_WORKERS)]
    },
    'task': {'type': 'worker', 'index': WORKER_NUM}
})




#import keras_model_to_estimator as kmte
#kmte.main("/tmp/kmtechkpt")

import  webtfkerasToEstimatorSimplemodel1 as  kmte
kmte.main1(model_dir='/home/cloud_user/model_dir')
