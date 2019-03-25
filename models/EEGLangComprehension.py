from bisect import bisect_left, bisect_right

import torch
from autokeras.supervised import DeepTaskSupervised, PortableDeepSupervised
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy, MSE
from autokeras.preprocessor import OneHotEncoder, DataTransformer, MultiTransformDataset, DataTransformerMlp
from autokeras.utils import pickle_to_file
from autokeras.constant import Constant
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class EEGLangComprehension(DeepTaskSupervised):

    def __init__(self, frequencies, frequency_component, **kwargs):
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
        return classification_loss

    def get_n_output_node(self):
        return 6

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
        #if self.y_encoder is None:
        #    self.y_encoder = OneHotEncoder()
        #    self.y_encoder.fit(y_train)
        #y_train = self.y_encoder.transform(y_train)
        return y_train

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableDeepSupervised(graph=self.cnn.best_model,
                                                y_encoder=self.y_encoder,
                                                data_transformer=self.data_transformer,
                                                path=self.path)
        pickle_to_file(portable_model, model_file_name)


class V1Transformer(DataTransformer):
    """
    Do Standardisation/z scoring for the x values
    """
    def __init__(self, data):
        super().__init__()
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform_train(self, data, targets=None, batch_size=None):
        data = torch.from_numpy((data - self.mean) / self.std)

        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)

        return DataLoader(MultiTransformDataset(data, targets, Compose([])), batch_size=batch_size, shuffle=True)

    def transform_test(self, data, targets=None, batch_size=None):
        self.transform_train(data, targets)
