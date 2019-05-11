import os
import json
NUM_WORKERS = 1
IP_ADDRS = ['localhost']
PORTS = [12345]

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['%s:%d' % (IP_ADDRS[w], PORTS[w]) for w in range(NUM_WORKERS)]
    },
    'task': {'type': 'worker', 'index': 0}
})

import webtfkerasToEstimatorSimplemodel1 as web
import tensorflow as tf
tf.__version__
train_data, train_label, validation_data, validation_label, test_data, test_label = web.data_preparation_moe()
predicted_indexes_moe=web.main1()
predicted_indexes_moe_array=[p['dense_1'] for p in predicted_indexes_moe]
import numpy as np
predicted_indexes_moe_list=np.array([p.tolist()[0] for p in predicted_indexes_moe_array])
test_label=np.array([t[0] for t in test_label])

predicted_indexes_moe = predicted_indexes_moe_list
#predicted_indexes_moe=predicted_indexes_moe.squeeze()
import pandas as pd
predicted_indexes_moe_df = pd.DataFrame({'predicted_index':predicted_indexes_moe,'actual_index':test_label})

error_in_index_moe=predicted_indexes_moe-test_label
import matplotlib.pyplot as plt

plt.plot([t[0] for t in train_label], error_in_index_moe)
