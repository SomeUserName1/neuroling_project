import os
from abc import abstractmethod, ABCMeta

import numpy as np
from keras.models import model_from_json

from util.BaseLogger import EmopyLogger


class AbstractNet(object, metaclass=ABCMeta):
    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size, steps_per_epoch,
                 epochs, preprocessor, logger, session):
        """
        initializes the basic class variables and the non-basic (e.g. different preprocessors) to None
        It is important to set the TAG of the net directly after calling super.init and esp. before initializing the
        logger
        Args:
            data_out_dir: directory where the data_collectors outputted to
            model_out_dir: directory where the weights, all logs and eventually visualizations of the weights are saved
            input_shape: the shape (width & height) of the input images
            learning_rate: the chosen learning rate
            batch_size: the amount of items per batch
            steps_per_epoch: the amounts of batches per epoch
            epochs: the amount of epochs
            preprocessor: A dedicated preprocessor to be set after calling super.init
            logger: The standard logger found int util/BaseLogger.py if None; by now there are no dedicated loggers
            session: either
        """
        self.net_type = net_type
        self.session = session
        self.logger = logger
        self.data_dir = data_out_dir
        self.model_out_dir = model_out_dir

        self.input_shape = input_shape
        self.preprocessor = preprocessor
        self.number_of_classes = self.preprocessor.classifier.get_num_class()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

        self.lr_decay = 0.99
        self.model = None

        if not os.path.exists(os.path.join(model_out_dir, self.net_type)):
            os.makedirs(os.path.join(model_out_dir, self.net_type))
        if logger is None:
            self.logger = EmopyLogger([os.path.join(model_out_dir, self.net_type, "%s.log" % self.net_type)])
        else:
            self.logger = logger

    def init_model(self, session):
        if session == "'train'":
            self.model = self.build()
        elif session == "'predict'":
            self.model = self.load_model()
        return self.model

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

    @abstractmethod
    def train(self):
        """


        """
        pass

    @abstractmethod
    def predict(self, faces):
        """

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
