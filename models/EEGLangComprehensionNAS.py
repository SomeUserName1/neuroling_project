from bisect import bisect_left, bisect_right

from autokeras.supervised import DeepTaskSupervised, PortableDeepSupervised
from autokeras.nn.loss_function import regression_loss
from autokeras.nn.metric import MSE
from autokeras.preprocessor import DataTransformerMlp
from autokeras.utils import pickle_to_file

import numpy as np


class EEGLangComprehensionNAS(DeepTaskSupervised):

    def __init__(self, path, frequencies, frequency_component, **kwargs):
        """

        :param electrodes: an array that defines the electrode names and their respective index
        :param frequencies: the available frequency components to map from Hz to the available frequency components
        :param electrodes_cluster: the electrodes that shall be used as input data to the classifier.
                If None all electrodes are used
        :param frequency_component: the frequencies that shall be used as input data to the classifier.
                If None all frequencies are used
        :param kwargs: args of the superclass
        """
        self.frequencies = frequencies
        self.frequency_component = frequency_component

        super().__init__(path=path, **kwargs)

    def fit(self, x, y, time_limit=None):
        x = self.preprocess(x)
        super(EEGLangComprehensionNAS, self).fit(x, y, time_limit)

    @property
    def metric(self):
        return MSE

    @property
    def loss(self):
            return regression_loss

    def get_n_output_node(self):
            return 1

    def init_transformer(self, x):
        if self.data_transformer is None:
            self.data_transformer = DataTransformerMlp(x)

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

    def transform_y(self, y_train):
            return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
            return output.flatten()

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableRegressor(graph=self.cnn.best_model,
                                                y_encoder=self.y_encoder,
                                                data_transformer=self.data_transformer,
                                                path=self.path,
                                                frequencies=self.frequencies,
                                                frequency_component=self.frequency_component)
        pickle_to_file(portable_model, model_file_name)


class PortableRegressor(PortableDeepSupervised):
    def __init__(self, path, frequencies, frequency_component, **kwargs):
        """

        :param electrodes: an array that defines the electrode names and their respective index
        :param frequencies: the available frequency components to map from Hz to the available frequency components
        :param electrodes_cluster: the electrodes that shall be used as input data to the classifier.
                If None all electrodes are used
        :param frequency_component: the frequencies that shall be used as input data to the classifier.
                If None all frequencies are used
        :param kwargs: args of the superclass
        """
        self.frequencies = frequencies
        self.frequency_component = frequency_component

        super().__init__(**kwargs)

    @property
    def metric(self):
        return MSE

    @property
    def loss(self):
        return regression_loss

    def get_n_output_node(self):
        return 1

    def init_transformer(self, x):
        if self.data_transformer is None:
            self.data_transformer = DataTransformerMlp(x)

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

    def transform_y(self, y_train):
        return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
        return output.flatten()