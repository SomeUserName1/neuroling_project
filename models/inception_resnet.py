import cv2
import keras.callbacks as cb
from keras import losses, optimizers, Input, Model
from keras.applications.inception_resnet_v2 import *
from keras.layers import Dropout, Dense
from keras.layers.advanced_activations import PReLU

from models.base import AbstractNet


class InceptionResNet(AbstractNet):
    def __init__(self, data_out_dir, model_out_dir, net_type, input_shape, learning_rate, batch_size, steps_per_epoch,
                 epochs,
                 preprocessor, logger, session):
        super(InceptionResNet, self).__init__(data_out_dir, model_out_dir, net_type, input_shape, learning_rate,
                                              batch_size,
                                              steps_per_epoch, epochs, preprocessor, logger, session)

        assert len(input_shape) == 3, "Input shape of neural network should be length of 3. e.g (64,64,1)"

        self.feature_extractors = ["image"]
        self.number_of_classes = self.preprocessor.classifier.get_num_class()
        self.model = super(InceptionResNet, self).init_model(session)

    def build(self):
        input_tensor = Input(shape=self.input_shape)
        incresnet = InceptionResNetV2(include_top=False, weights=None, classes=7, input_shape=self.input_shape,
                                      pooling='avg', input_tensor=input_tensor)
        res_out = incresnet(input_tensor)
        x = Dense(4096)(res_out)
        x = PReLU()(x)
        x = Dropout(0.382)(x)
        x = Dense(4024)(x)
        x = PReLU()(x)
        x = Dense(2048)(x)
        x = PReLU()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024)(x)
        x = PReLU()(x)
        x = Dense(self.number_of_classes, activation='softmax', name='predictions')(x)
        self.model = Model(input_tensor, x)

        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.Adam(self.learning_rate),
                           metrics=['accuracy'])

        return self.model

    def train(self):
        dir = self.model_out_dir + '/' + self.net_type + '/'
        tb = cb.TensorBoard(log_dir=dir + '/tensorboard-logs',
                            batch_size=self.batch_size)
        checkpoint = cb.ModelCheckpoint(dir + '/weights.h5', mode='min', save_best_only=True,
                                        save_weights_only=False, verbose=1)
        lr_decay = cb.LearningRateScheduler(schedule=lambda epoch: self.learning_rate * (self.lr_decay ** epoch))

        self.preprocessor = self.preprocessor(self.data_dir)
        self.model.fit_generator(self.preprocessor.flow(), steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 validation_data=([self.preprocessor.test_images, self.preprocessor.test_dpoints,
                                                   self.preprocessor.dpointsDists, self.preprocessor.dpointsAngles],
                                                  self.preprocessor.test_image_emotions),
                                 callbacks=[tb, checkpoint, lr_decay])
        score = self.model.evaluate(
            [self.preprocessor.test_images, self.preprocessor.test_dpoints, self.preprocessor.dpointsDists,
             self.preprocessor.dpointsAngles], self.preprocessor.test_image_emotions)
        self.save_model()
        self.logger.log_model(self.net_type, score, self.model)

    def predict(self, face):
        """

        Args:
            face:

        Returns:

        """
        assert face.shape == self.input_shape, "Face image size should be " + str(self.input_shape)
        face = face.reshape(1, 64, 64)

        cv2.imshow("img", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        emotions = self.model.predict(face)[0]
        return emotions
