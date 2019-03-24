from autokeras.supervised import DeepTaskSupervised
from autokeras.constant import Constant
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy, MSE
from autokeras.preprocessor import OneHotEncoder, DataTransformer


class EEGLangComprehension(DeepTaskSupervised):
    @property
    def metric(self):
        return Accuracy

    @property
    def loss(self):
        return classification_loss

    def get_n_output_node(self):
        return 6

    def init_transformer(self, x):
        pass

    def preprocess(self, x):
        pass


class V1Transformer(DataTransformer):
    def transform_train(self, data, targets=None, batch_size=None):
        pass

    def transform_test(self, data, targets=None, batch_size=None):
        pass

