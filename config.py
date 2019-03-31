# coding=utf-8
import os

# Constants
ALL_ELECTRODES = ['AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4',
                  'CP5', 'CP6',
                  'CPZ', 'CZ', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5',
                  'FC6', 'FCZ', 'FP1',
                  'FP2', 'FT10', 'FT7', 'FT8', 'FT9', 'FZ', 'O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
                  'P7', 'P8',
                  'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ', 'T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9']
CLUSTER = ['AFZ', 'C2', 'C4', 'CP4', 'CP6', 'F1']

# Directory setup
DATA_SET_DIR = os.sep.join(['C:', 'Users', 'Fabi', 'ownCloud', 'workspace', 'uni', '7', 'neuroling',
                            'neuroling_project', 'data', 'v1'])
MODEL_OUT_DIR = os.sep.join(['C:', 'Users', 'Fabi', 'ownCloud', 'workspace', 'uni', '7', 'neuroling',
                             'neuroling_project', 'models', 'weights'])

# Pre-processing setup
ELECTRODES = ALL_ELECTRODES
FREQUENCY = 10


# Training & Testing Setup
VERBOSE = True  # how much to print
SEARCH_TIME = 10 * 60  # in s
LEARNING_RATE = 2e-2    # best practice: sth between 1e-1 and 1e-6
BATCH_SIZE = 128  # Batch sized used for training, i.e. how many images per batch
EPOCHS = 10000      # how many times to look at the data set. Normally sth. between 10 and 100. Early stopping is
# implemented so only a fraction of the value here is executed