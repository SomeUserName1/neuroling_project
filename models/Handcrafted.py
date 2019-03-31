# coding=utf-8
from keras import Input, Model
from keras.layers import Flatten, Dense, Conv1D
import pydot
from models.base import AbstractNet


class MediumConv1DNet(AbstractNet):
    def __init__(self, model_out_dir, frequency, electrodes):
        """
        Args:
            model_out_dir: directory to save model to
            frequency: the frequency to be used (shape[1])
            electrodes: the electrodes to be used (shape[0])
        """
        super(MediumConv1DNet, self).__init__('MediumConv1DNet', model_out_dir, frequency, electrodes)

    def build(self):
        """
        constructs a model using 2 conv1D layers and 3 Dense layers
        Returns:
            the constructed model
        """
        a = Input(self.input_shape)
        x = Conv1D(self.input_shape[1], self.input_shape[0], activation='relu', name='0c0')(a)
        x = Flatten()(x)
        x = Dense(32, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='0fc0')(x)
        x = Dense(5, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='0fc1')(x)
        b = Dense(1, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='0fc2')(x)
        self.model = Model(a, b)
        super().build()


class SmallDenseNet(AbstractNet):
    def __init__(self, model_out_dir, frequency, electrodes):
        """
        Args:
            model_out_dir: directory to save model to
            frequency: the frequency to be used (shape[1])
            electrodes: the electrodes to be used (shape[0])
        """
        super(SmallDenseNet, self).__init__('SmallDenseNet', model_out_dir, frequency, electrodes)

    def build(self):
        """
        constructs a model using 2 conv1D layers and 3 Dense layers
        Returns:
            the constructed model
        """
        a = Input(self.input_shape)
        x = Flatten()(a)
        b = Dense(1, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='1fc2')(x)
        self.model = Model(a, b)
        super().build()


class WideDenseNet(AbstractNet):
    def __init__(self, model_out_dir, frequency, electrodes):
        """
        Args:
            model_out_dir: directory to save model to
            frequency: the frequency to be used (shape[1])
            electrodes: the electrodes to be used (shape[0])
        """
        super(WideDenseNet, self).__init__('WideDenseNet', model_out_dir, frequency, electrodes)

    def build(self):
        """
        constructs a model using 2 conv1D layers and 3 Dense layers
        Returns:
            the constructed model
        """
        a = Input(self.input_shape)
        x = Flatten()(a)
        x = Dense(64, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='0fc0')(x)
        b = Dense(1, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='0fc2')(x)
        self.model = Model(a, b)
        super().build()


class DeepDenseNet(AbstractNet):
    def __init__(self, model_out_dir, frequency, electrodes):
        """
        Args:
            model_out_dir: directory to save model to
            frequency: the frequency to be used (shape[1])
            electrodes: the electrodes to be used (shape[0])
        """
        super(DeepDenseNet, self).__init__('DeepDenseNet', model_out_dir, frequency, electrodes)

    def build(self):
        """
        constructs a model using 4 Dense layers
        Returns:
            the constructed model
        """
        a = Input(self.input_shape)
        x = Flatten()(a)
        x = Dense(8, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='3fc0')(x)
        x = Dense(4, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='3fc1')(x)
        x = Dense(2, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='3fc2')(x)
        b = Dense(1, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros', name='3fc3')(x)
        self.model = Model(a, b)
        super().build()