from __future__ import print_function

import os
import sys
import time

import numpy as np
from keras.utils import print_summary, plot_model

from config import DATA_SET_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE


class Logger(object):
    """
    Basic keras model logger
    """
    def __init__(self, output_dir, net_type):
        """
        Args:
            output_dir: location where all model weights are saved
            net_type: identifier string for the model, name of the subfolder to log to
        """
        self.net_type = net_type
        self.output_dir = os.path.join(output_dir, net_type)
        self.output_file = os.path.join(self.output_dir, "%s.log" % self.net_type)

    def log(self, string):
        """
            log a single string
        Args:
            string: string to log
        """
        if self.output_file == sys.stdout:
            print(string)
        elif type(self.output_file) == str:
            with open(self.output_file, "a+") as out_file:
                out_file.write(string + "\n")

    def log_model(self, score, model):
        """
        Logs the performance and architecture of a model
        Args:
            score:
            model:
        """
        model_number = np.fromfile(os.sep.join([self.output_dir, "model_number.txt"]), dtype=int)
        model_file_name = self.net_type + "-" + str(model_number[0] - 1)

        self.log("=========================================Start of Log==============================================")
        self.log("Trained model " + model_file_name + ".json")
        self.log(time.strftime("%A %B %d,%Y %I:%M%p"))
        self.log("Dataset dir: " + DATA_SET_DIR)
        print_summary(model, print_fn=self.log)
        self.log("___________Parameters:______________")
        self.log("Batch size    : " + str(BATCH_SIZE))
        self.log("Epochs       : " + str(EPOCHS))
        self.log("Learning rate : " + str(LEARNING_RATE))
        self.log("___________Metrics___________________")
        self.log("Loss          : " + str(score[0]))
        self.log("MSE      : " + str(score[1]))
        self.log("=========================================End of Log=================================================")
        self.log("====================================================================================================")
        self.log("----------------------------------------------------------------------------------------------------")
        plot_model(model, show_shapes=True, to_file=os.sep.join([self.output_dir, model_file_name + ".png"]))
