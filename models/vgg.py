# coding=utf-8
from keras import Input, Model, losses, optimizers
from keras.layers import Flatten, Dense, Conv1D
from keras_applications.vgg16 import VGG16
from models.base import AbstractNet

# TODO

class EEGLangComprehensionVGG(AbstractNet):
    def __init__(self, frequencies, frequency_component, model_out_dir, learning_rate, batch_size, steps_per_epoch, epochs):
        super(EEGLangComprehensionVGG, self).__init__(frequencies, frequency_component, 'vgg', model_out_dir, learning_rate, batch_size,
                                              steps_per_epoch, epochs)
        self.TAG = "vgg"



