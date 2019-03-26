from bisect import bisect_left, bisect_right

import torch
from autokeras.supervised import DeepTaskSupervised, PortableDeepSupervised
from autokeras.nn.loss_function import regression_loss, classification_loss
from autokeras.nn.metric import MSE, Accuracy
from autokeras.preprocessor import DataTransformerMlp, OneHotEncoder
from autokeras.utils import pickle_to_file
import numpy as np


class EEGLangComprehensionNAS(DeepTaskSupervised):

    def __init__(self, regression, frequencies, frequency_component, **kwargs):
        """

        :param electrodes: an array that defines the electrode names and their respective index
        :param frequencies: the available frequency components to map from Hz to the available frequency components
        :param electrodes_cluster: the electrodes that shall be used as input data to the classifier.
                If None all electrodes are used
        :param frequency_component: the frequencies that shall be used as input data to the classifier.
                If None all frequencies are used
        :param kwargs: args of the superclass
        """
        self.regression = regression
        self.frequencies = frequencies
        self.frequency_component = frequency_component
        super().__init__(**kwargs)

    @property
    def metric(self):
        metric = MSE
        if not self.regression:
            metric = Accuracy
        return metric

    @property
    def loss(self):
        loss = regression_loss
        if not self.regression:
            loss = classification_loss
        return loss

    def get_n_output_node(self):
        n_out = 1
        if not self.regression:
            n_out = self.y_encoder.n_classes
        return n_out

    def init_transformer(self, x):
        if self.data_transformer is None:
            self.data_transformer = DataTransformerMlp(x)

    def preprocess(self, x):
        preprocessed_x = []
        if self.frequency_component is not None:
            if self.frequency_component < 1 or self.frequency_component > 50:
                raise Exception("Select a frequency between 1 and 50 Hz!")
            else:
                preprocessed_x = np.squeeze(x[:, :, self.frequency_lookup()])

        return preprocessed_x

    def frequency_lookup(self):
        lower = self.frequency_component - 1
        upper = self.frequency_component + 1

        idx_lower = bisect_right(self.frequencies, lower)
        idx_upper = bisect_left(self.frequencies, upper)

        return [range(idx_lower, idx_upper + 1)]

    def transform_y(self, y_train):
        transformed_y = y_train.flatten().reshape(len(y_train), 1)
        if not self.regression:
            self.y_encoder = OneHotEncoder()
            self.y_encoder.fit(y_train)
            transformed_y = self.y_encoder.transform(y_train)

        return transformed_y

    def inverse_transform_y(self, output):
        return output.flatten()

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableDeepSupervised(graph=self.cnn.best_model,
                                                y_encoder=self.y_encoder,
                                                data_transformer=self.data_transformer,
                                                path=self.path)
        pickle_to_file(portable_model, model_file_name)
