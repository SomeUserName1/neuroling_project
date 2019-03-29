from autokeras.nn.loss_function import regression_loss
from autokeras.nn.metric import MSE
from autokeras.preprocessor import DataTransformer, MultiTransformDataset
from autokeras.supervised import DeepTaskSupervised, PortableDeepSupervised
from autokeras.utils import pickle_to_file
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


class NAS(DeepTaskSupervised):
    """
    Class that shall perform a neural architecture search for the general regression problem.
    """
    def __init__(self, training_time=10 * 60, **kwargs):
        """
        initializes the regressor neural architecture search
        Args:
            training_time: how long the nas should run
            kwargs: further args derived from base class
        """
        self.time_limit = training_time
        super().__init__(**kwargs)

    def fit(self, x, y, time_limit=None):
        """
        finds the best model for a given data set
        Args:
            x: the data to regress from
            y: the correct values that are to be approximated by regression
            time_limit:
        """
        super().fit(x, y, time_limit=self.time_limit)

    def preprocess(self, x):
        """
        needed from superclass, does nothing but returning the argument directly
        Args:
            x: training data
        Returns: unmodified training data

        """
        return x

    @property
    def metric(self):
        """
        the metric function to evaluate the regressor. As the problem is regression and only the mean squared error is
        supported we use the mean squared error
        Returns:
                the metric function of the regressor
        """
        return MSE

    @property
    def loss(self):
        """
        the loss function to evaluate the regressor. As the problem is regression we use the standard regression loss of
        scipy
        Returns:
                the loss function of the regressor
            """
        return regression_loss

    def get_n_output_node(self):
        """
        Number of output neurons. As we are training a one valued regression we need one output neuron
            Returns:
                the number of output neurons
            """
        return 1

    def init_transformer(self, x):
        """
        Initializes the class that packs the data into appropriate format for torch/tf
        Args:
            x: the data
        """
        if self.data_transformer is None:
            self.data_transformer = DataTransformerEmpty()

    def transform_y(self, y_train):
        """
        Transforms the actual values of the regression to a (len, 1) array
            Args:
                y_train:
                    the actual values to regress
            Returns:
                acutal values in shape (len, 1)
            """
        return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
        """
            undoes the changes applied in transform_y
            Args:
                output:
                    what the regressor predicted
            Returns:
                the predicted value in the same format as the input actual values were
            """
        return output.flatten()

    def export_autokeras_model(self, model_file_name):
        """
        Creates and Exports the AutoKeras model to the given filename.
        Args:
            model_file_name: the name of the model to safe
        """
        portable_model = PortableRegressor(graph=self.cnn.best_model,
                                           y_encoder=self.y_encoder,
                                           data_transformer=self.data_transformer,
                                           path=self.path)
        pickle_to_file(portable_model, model_file_name)


class PortableRegressor(PortableDeepSupervised):
    """
    Wrapper class to export a regressor to a keras model
    """

    def preprocess(self, x):
        """
            Does nothing, neccessary due to inheritance
        Args:
            x: input data

        Returns:
            unmodified input data
        """
        return x

    @property
    def metric(self):
        """
        the metric function to evaluate the regressor. As the problem is regression and only the mean squared error is
        supported we use the mean squared error
        Returns:
                the metric function of the regressor
        """
        return MSE

    @property
    def loss(self):
        """
        the loss function to evaluate the regressor. As the problem is regression we use the standard regression loss of
        scipy
        Returns:
                the loss function of the regressor
            """
        return regression_loss

    def init_transformer(self, x):
        """
        Initializes the class that packs the data into appropriate format for torch/tf
        Args:
            x: the data
        """
        if self.data_transformer is None:
            self.data_transformer = DataTransformerEmpty()

    def transform_y(self, y_train):
        """
        Transforms the actual values of the regression to a (len, 1) array
            Args:
                y_train:
                    the actual values to regress
            Returns:
                acutal values in shape (len, 1)
            """
        return y_train.flatten().reshape(len(y_train), 1)

    def inverse_transform_y(self, output):
        """
            undoes the changes applied in transform_y
            Args:
                output:
                    what the regressor predicted
            Returns:
                the predicted value in the same format as the input actual values were
            """
        return output.flatten()


class DataTransformerEmpty(DataTransformer):
    """
    DataTransformer that only transforms into the framework defined input tensor shape and packs the values into a
     DataLoader
    """

    def transform_train(self, data, targets=None, batch_size=32):
        """
            transforms to valid format and returns the data loader wrapped data set
        Args:
            data: data set to be fed into the regressor
            targets: labels to be fed into the regressor
            batch_size: the batch size to use

        Returns:
            The transformed data set wrapped in a DataLoader
        """
        dataset = self._transform(data, targets)

        return DataLoader(dataset, batch_size=32, shuffle=False)

    def transform_test(self, data, target=None, batch_size=None):
        """
            transforms to valid format and returns the data loader wrapped data set
        Args:
            data: data set to be fed into the regressor
            target: labels to be fed into the regressor
            batch_size: the batch size to use

        Returns:
            The transformed data set wrapped in a DataLoader
        """
        return self.transform_train(data, targets=target, batch_size=batch_size)

    @staticmethod
    def _transform(data, targets, compose_list=None):
        """
            Transposes the data set as required by the underlying framework (tf/torch)
        Args:
            compose_list: list of transformations to apply to the data
            data: data to be transformed
            targets: labels

        Returns:
            transposed data set
        """
        if compose_list is None:
            compose_list = []
        args = [0, len(data.shape) - 1] + list(range(1, len(data.shape) - 1))
        data = Tensor(data.transpose(*args))
        data_transforms = Compose(compose_list)
        return MultiTransformDataset(data, targets, data_transforms)
