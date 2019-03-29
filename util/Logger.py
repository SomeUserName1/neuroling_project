from __future__ import print_function

import os
import sys
import time

import numpy as np
from keras.utils import print_summary, plot_model

from config import MODEL_OUT_DIR, DATA_SET_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE


class Logger(object):
    """
    Basic keras model logger
    """

    def __init__(self, output_files):
        """
        Args:
            output_files: location to log to
        """
        self.output_files = output_files

    def log(self, string):
        """
            log a single string
        Args:
            string: string to log
        """
        for f in self.output_files:
            if f == sys.stdout:
                print(string)
            elif type(f) == str:
                with open(f, "a+") as out_file:
                    out_file.write(string + "\n")

    def log_model(self, models_local_folder, score, model):
        """
        Logs the performance and architecture of a model
        Args:
            models_local_folder:
            score:
        """
        model_number = np.fromfile(os.path.join(MODEL_OUT_DIR, models_local_folder, "model_number.txt"), dtype=int)
        model_file_name = models_local_folder + "-" + str(model_number[0] - 1)

        self.log("=========================================Start of Log==============================================")
        self.log("Trained model " + model_file_name + ".json")
        self.log(time.strftime("%A %B %d,%Y %I:%M%p"))
        self.log("Dataset dir: " + DATA_SET_DIR)
        print_summary(model, print_fn=self.log)
        self.log("Parameters")
        self.log("_______________________________________")
        self.log("Batch size    : " + str(BATCH_SIZE))
        self.log("Epochs       : " + str(EPOCHS))
        self.log("Learning rate : " + str(LEARNING_RATE))
        self.log("_______________________________________")
        self.log("Loss          : " + str(score[0]))
        self.log("MSE      : " + str(score[1]))
        self.log("=========================================End of Log=================================================")
        self.log("====================================================================================================")
        self.log("----------------------------------------------------------------------------------------------------")
        plot_model(model, show_shapes=True, to_file=os.sep.join([MODEL_OUT_DIR, models_local_folder, model_file_name +
                                                                 ".png"]))
