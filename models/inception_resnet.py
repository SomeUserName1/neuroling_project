import cv2
import keras.callbacks as cb
from keras import losses, optimizers, Input, Model
from keras.applications.inception_resnet_v2 import *
from keras.layers import Dropout, Dense
from keras.layers.advanced_activations import PReLU

from models.base import AbstractNet

# TODO

class EEGLangComprehensionInceptionResNet(AbstractNet):
    def __init__(self, model_out_dir, net_type, learning_rate, batch_size, steps_per_epoch, epochs, session):
        super(EEGLangComprehensionInceptionResNet, self).__init__(model_out_dir, net_type, learning_rate, batch_size,
                                              steps_per_epoch, epochs, session)

    def build(self):
        input_tensor = Input(shape=self.input_shape)
        incresnet = InceptionResNetV2(include_top=False, weights=None, classes=7, input_shape=self.input_shape,
                                      pooling='avg', input_tensor=input_tensor)
        res_out = incresnet(input_tensor)
        x = Dense(6)(res_out)
        x = PReLU()(x)
        x = Dense(4024)(x)
        x = PReLU()(x)
        x = Dense(2048)(x)
        x = PReLU()(x)
        x = Dense(1024)(x)
        x = PReLU()(x)
        x = Dense(self.number_of_classes, activation='softmax', name='predictions')(x)
        self.model = Model(input_tensor, x)

        self.model.compile(loss=losses.mean_squared_error,
                           optimizer=optimizers.Adam(self.learning_rate),
                           metrics=losses.mean_squared_error)

        return self.model
