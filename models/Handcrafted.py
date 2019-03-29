# coding=utf-8
from keras import Input, Model, optimizers
from keras.layers import Flatten, Dense, Conv1D

from models.base import AbstractNet


class MediumConv1DNet(AbstractNet):
    def __init__(self, model_out_dir, frequency, electrodes):
        """
        Args:
            model_out_dir: directory to save model to
            frequency: the frequency to be used (shape[1])
            electrodes: the electrodes to be used (shape[9])
        """
        super(MediumConv1DNet, self).__init__('MediumConv1DNet', model_out_dir, frequency, electrodes)

    def build(self):
        """
        constructs a model using 2 conv1D layers and 3 Dense layers
        Returns:
            the constructed model
        """
        a = Input(self.input_shape)
        x = Conv1D(16, 5, activation='relu', name='c0')(a)
        x = Conv1D(8, 1, activation='relu', name='c1')(x)
        x = Flatten()(x)
        x = Dense(4, activation='sigmoid', kernel_initializer='normal', name='fc0')(x)
        x = Dense(2, activation='sigmoid', kernel_initializer='normal', name='fc1')(x)
        b = Dense(1, activation='sigmoid', kernel_initializer='normal', name='fc2')(x)
        self.model = Model(a, b)
        super().build()


class SmallDenseNet(AbstractNet):
    def __init__(self, model_out_dir, frequency, electrodes):
        """
        Args:
            model_out_dir: directory to save model to
            frequency: the frequency to be used (shape[1])
            electrodes: the electrodes to be used (shape[9])
        """
        super(SmallDenseNet, self).__init__('SmallDenseNet', model_out_dir, frequency, electrodes)

    def build(self):
        """
        constructs a model using 1 Dense layer
        Returns:
            the constructed model
        """
        a = Input(self.input_shape)
        x = Flatten()(a)
        b = Dense(1, activation='sigmoid', kernel_initializer='normal', name='fc2')(x)
        self.model = Model(a, b)
        super().build()


class WideDenseNet(AbstractNet):
    def __init__(self, model_out_dir, frequency, electrodes):
        """
        Args:
            model_out_dir: directory to save model to
            frequency: the frequency to be used (shape[1])
            electrodes: the electrodes to be used (shape[9])
        """
        super(WideDenseNet, self).__init__('WideDenseNet', model_out_dir, frequency, electrodes)

    def build(self):
        """
        constructs a model using 2 Dense layers
        Returns:
            the constructed model
        """
        a = Input(self.input_shape)
        x = Flatten()(a)
        x = Dense(64, activation='sigmoid', kernel_initializer='normal', name='fc0')(x)
        b = Dense(1, activation='sigmoid', kernel_initializer='normal', name='fc2')(x)
        self.model = Model(a, b)
        super().build()


class DeepDenseNet(AbstractNet):
    def __init__(self, model_out_dir, frequency, electrodes):
        """
        Args:
            model_out_dir: directory to save model to
            frequency: the frequency to be used (shape[1])
            electrodes: the electrodes to be used (shape[9])
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
        x = Dense(8, activation='sigmoid', kernel_initializer='normal', name='fc0')(x)
        x = Dense(4, activation='sigmoid', kernel_initializer='normal', name='fc1')(x)
        x = Dense(2, activation='sigmoid', kernel_initializer='normal', name='fc2')(x)
        b = Dense(1, activation='sigmoid', kernel_initializer='normal', name='fc3')(x)
        self.model = Model(a, b)
        super().build()