from autokeras.supervised import DeepTaskSupervised, PortableDeepSupervised
from autokeras.nn.loss_function import regression_loss
from autokeras.nn.metric import MSE
from autokeras.preprocessor import DataTransformerMlp, DataTransformer, MultiTransformDataset
from autokeras.utils import pickle_to_file
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch import Tensor


class NAS(DeepTaskSupervised):
    """
    """
    def __init__(self, training_time=10 * 60, **kwargs):
        self.time_limit = training_time
        super().__init__(**kwargs)

    def fit(self, x, y, time_limit=None):
        super().fit(x, y, time_limit=self.time_limit)

    def preprocess(self, x):
        return x

    @property
    def metric(self):
        """

        Returns:

        """
        return MSE

    @property
    def loss(self):
        """

            Returns:

            """
        return regression_loss

    def get_n_output_node(self):
        """

            Returns:

            """
        return 1

    def init_transformer(self, x):
        """

        Args:
            x:
        """
        if self.data_transformer is None:
            self.data_transformer = DataTransformerEmpty()

    def transform_y(self, y_train):
        """

            Args:
                y_train:

            Returns:

            """
        return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
        """

            Args:
                output:

            Returns:

            """
        return output.flatten()

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableRegressor(graph=self.cnn.best_model,
                                           y_encoder=self.y_encoder,
                                           data_transformer=self.data_transformer,
                                           path=self.path)
        pickle_to_file(portable_model, model_file_name)


class PortableRegressor(PortableDeepSupervised):
    """
    """

    def __init__(self, **kwargs):
        """

        Args:
            electrodes: an array that defines the electrode names and their respective index
            frequencies: the available frequency components to map from Hz to the available frequency components
            electrodes_cluster: the electrodes that shall be used as input data to the classifier.
                    If None all electrodes are used
            frequency_component: the frequencies that shall be used as input data to the classifier.
                    If None all frequencies are used
            kwargs: args of the superclass
        """
        super().__init__(**kwargs)

    def preprocess(self, x):
        return x

    @property
    def metric(self):
        """

        Returns:

        """
        return MSE

    @property
    def loss(self):
        """

        Returns:

        """
        return regression_loss

    def init_transformer(self, x):
        """

        Args:
            x:
        """
        if self.data_transformer is None:
            self.data_transformer = DataTransformerMlp(x)

    def transform_y(self, y_train):
        """

        Args:
            y_train:

        Returns:

        """
        return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
        """

        Args:
            output:

        Returns:

        """
        return output.flatten()


class DataTransformerEmpty(DataTransformer):

    def transform_train(self, data, targets=None, batch_size=32):
        dataset = self._transform([], data, targets)

        return DataLoader(dataset, batch_size=32, shuffle=False)

    def transform_test(self, data, target=None, batch_size=None):
        return self.transform_train(data, targets=target, batch_size=batch_size)
    @staticmethod
    def _transform(compose_list, data, targets):
        args = [0, len(data.shape) - 1] + list(range(1, len(data.shape) - 1))
        data = Tensor(data.transpose(*args))
        data_transforms = Compose(compose_list)
        return MultiTransformDataset(data, targets, data_transforms)