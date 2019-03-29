import os
from abc import abstractmethod, ABCMeta

import numpy as np
from keras import optimizers
from keras.models import model_from_json

from util.Logger import Logger


class AbstractNet(object, metaclass=ABCMeta):
    def __init__(self, net_type, model_out_dir, frequency, electrodes, learning_rate=0.0001, batch_size=32,
                 epochs=30):
        """
        initializes the basic class variables
        Args:
            learning_rate: the chosen learning rate
            batch_size: the amount of items per batch
            epochs: the amount of epochs
        """
        if frequency is None:
            self.input_shape = (len(electrodes), 101)
        else:
            self.input_shape = (len(electrodes), 5)
        self.net_type = net_type
        self.model_out_dir = model_out_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model = None
        self.history = None

        self.logger = Logger([os.path.join(model_out_dir, self.net_type, "%s.log" % self.net_type)])

        if not os.path.exists(os.path.join(model_out_dir, self.net_type)):
            os.makedirs(os.path.join(model_out_dir, self.net_type))

    @abstractmethod
    def build(self):
        """
        Build neural network model

        Returns:
            neural network model
        """
        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizers.RMSprop(self.learning_rate),
                           metrics=['mean_absolute_error', 'mean_squared_error'])
        self.model.summary()

    def fit(self, x, y):
        if self.model is None:
            self.build()
        self.history = self.model.fit(x, y, epochs=self.epochs, validation_split=0.25, batch_size=self.batch_size,
                                      verbose=True)

    def evaluate(self, x, y):
        """
            evaluates the neural network model
        Args:
            x: data set
            y: labels
        """
        if self.model is None:
            self.build()
            self.fit(x, y)
        # TODO get them running
        # tb = cb.TensorBoard(log_dir=os.sep.join([self.model_out_dir, self.net_type, 'tensorboard-logs']),
        #                    write_graph=True, write_images=True)
        # cp = cb.ModelCheckpoint(os.sep.join([self.model_out_dir, self.net_type, 'weights.h5']), save_best_only=True,
        #                        save_weights_only=False, verbose=1)
        # es = cb.EarlyStopping(monitor='val_loss', min_delta=0.007, restore_best_weights=True)
        # cb_list = [cp] #, tb, es]

        score = self.model.evaluate(x, y, batch_size=self.batch_size)  # , callbacks=cb_list))
        self.logger.log_model(self.net_type, score, self.model)
        self.save_model()

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
