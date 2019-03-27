import os
from abc import abstractmethod, ABCMeta
from bisect import bisect_right, bisect_left

import keras.callbacks as cb
import numpy as np
from keras.models import model_from_json

from util.BaseLogger import Logger

# TODO

class AbstractNet(object, metaclass=ABCMeta):
    def __init__(self, frequencies, frequency_component, net_type, model_out_dir, learning_rate, batch_size, steps_per_epoch,
                 epochs):
        """
        initializes the basic class variables and the non-basic (e.g. different preprocessors) to None
        It is important to set the net type/name in the loffer of the net directly after calling super.init and esp.
         before initializing the logger
        Args:
            learning_rate: the chosen learning rate
            batch_size: the amount of items per batch
            steps_per_epoch: the amounts of batches per epoch
            epochs: the amount of epochs

        """
        self.net_type = net_type
        self.logger = Logger([os.path.join(model_out_dir, self.net_type, "%s.log" % self.net_type)])
        self.model_out_dir = model_out_dir
        self.input_shape = (6,5)
        self.number_of_classes = 1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

        self.lr_decay = 0.99
        self.model = None

        self.frequency_component = frequency_component
        self.frequencies = frequencies

        if not os.path.exists(os.path.join(model_out_dir, self.net_type)):
            os.makedirs(os.path.join(model_out_dir, self.net_type))

    @abstractmethod
    def build(self):
        """
        Build neural network model

        Returns
        -------
        keras.models.Model :
            neural network model
        """
        pass

    def save_model(self):
        """
        Saves NeuralNet model. The naming convention is for json and h5 files is,
        `/path-to-models/model-local-folder-model-number.json` and
        `/path-to-models/model-local-folder-model-number.h5` respectively.
        This method also increments model_number inside "model_number.txt" file.
        """
        if not os.path.exists(self.model_out_dir):
            os.makedirs(self.model_out_dir)
        out_dir = os.path.join(self.model_out_dir, self.net_type)
        if not os.path.exists(out_dir):
            os.makedirs(os.path.join(out_dir))
        if not os.path.exists(os.path.join(out_dir, "model_number.txt")):
            model_number = np.array([0])
        else:
            model_number = np.fromfile(os.path.join(out_dir, "model_number.txt"),
                                       dtype=int)
        model_file_name = self.net_type + "-" + str(model_number[0])
        with open(os.path.join(self.model_out_dir, self.net_type, model_file_name + ".json"), "a+") as jfile:
            jfile.write(self.model.to_json())
        self.model.save_weights(os.path.join(out_dir, model_file_name + ".h5"))
        model_number[0] += 1
        model_number.tofile(os.path.join(out_dir, "model_number.txt"))

    def load_model(self):
        """
        loads a pre-trained model
        Returns: the loaded model

        """
        if not os.path.exists(self.model_out_dir):
            os.makedirs(self.model_out_dir)
        out_dir = os.path.join(self.model_out_dir, self.net_type)
        if not os.path.exists(out_dir):
            os.makedirs(os.path.join(out_dir))
        if not os.path.exists(os.path.join(out_dir, "model_number.txt")):
            raise Exception("no models created by now, run training first")
        else:
            model_number = np.fromfile(os.path.join(out_dir, "model_number.txt"),
                                       dtype=int) - 1

        model_file_name = self.net_type + "-" + str(model_number[0])
        path = os.path.join(self.model_out_dir, self.net_type, model_file_name)

        with open(path + ".json") as model_file:
            model = model_from_json(model_file.read())

        model.load_weights(path + ".h5")
        print("loaded model")
        return model

    def preprocess(self, x):
        if self.frequency_component is not None:
            if self.frequency_component < 1 or self.frequency_component > 50:
                raise Exception("Select a frequency between 1 and 50 Hz!")
            else:
                return np.squeeze(x[:, :, self.frequency_lookup()])
        else:
            return x

    def frequency_lookup(self):
        lower = self.frequency_component - 1
        upper = self.frequency_component + 1

        idx_lower = bisect_right(self.frequencies, lower)
        idx_upper = bisect_left(self.frequencies, upper)

        return [range(idx_lower, idx_upper + 1)]