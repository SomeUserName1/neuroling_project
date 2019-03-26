# coding=utf-8
from keras.layers import Flatten, Dense
from keras_applications.vgg16 import VGG16
from models.base import AbstractNet


# TODO test
class VGGFaceEmopyNet(AbstractNet):
    """
    Class for implementation of EmoPy using VGG Face Net as base
    according to http://www.robots.ox.ac.uk/%7Evgg/software/vgg_face/.
    """

    def train(self):
        pass

    def predict(self, faces):
        pass

    def __init__(self, data_out_dir, model_out_dir, input_shape, learning_rate, batch_size, steps_per_epoch, epochs,
                 preprocessor=None, logger=None, session='train', post_processor=None):
        super(VGGFaceEmopyNet, self).__init__(data_out_dir, model_out_dir, input_shape, learning_rate, batch_size,
                                              steps_per_epoch, epochs,
                                              preprocessor=None, logger=None, session='train')
        self.TAG = "vgg"
        self.max_sequence_length = 10
        self.postProcessor = post_processor
        self.feature_extractors = ['image']
        self.number_of_class = self.preprocessor.classifier.get_num_class()
        super(VGGFaceEmopyNet, self).init_logger(self.logger, self.model_out_dir, self.TAG)
        super(VGGFaceEmopyNet, self).init_model(self.session)

    def build(self):
        """

        Returns:
            An instance of the EmoPy VGG Face model
        """
        # x = VGGFace(include_top=False, input_shape=self.input_shape)
        vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(32, activation='relu', name='fc6')(x)
        x = Dense(32, activation='relu', name='fc7')(x)
        print("VGG")
        x.summary()
        return vgg_model
