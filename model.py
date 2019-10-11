import tensorflow as tf
from utils import *

keras = tf.keras

BatchNorm = keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
DConv = keras.layers.Conv2DTranspose


class AutoEncoder(keras.models.Model):
    def __init__(self, config, input_shape, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.config = load_config(config)

        self.act_fn = keras.layers.LeakyReLU()
        self.kernel_size = self.config['kernel_size']

        self.conv = [
            Conv2D(filters=32, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None,
                   input_shape=input_shape),
            BatchNorm(),
            self.act_fn,
            Conv2D(filters=64, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            BatchNorm(),
            self.act_fn,
            Conv2D(filters=128, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            BatchNorm(),
            self.act_fn,
            Conv2D(filters=256, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            BatchNorm(),
            self.act_fn,
            Conv2D(filters=512, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            BatchNorm(),
            self.act_fn,
            #             keras.layers.Flatten(),
            #             Dense(units=output_size, activation=None),
            #             self.act_fn
        ]

        self.dconv = [
            # 16
            DConv(filters=512, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            BatchNorm(),
            self.act_fn,
            # 32
            DConv(filters=256, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            BatchNorm(),
            self.act_fn,
            # 64
            DConv(filters=128, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            BatchNorm(),
            self.act_fn,

            DConv(filters=64, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            BatchNorm(),
            self.act_fn,

            DConv(filters=1, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
            keras.layers.ReLU(max_value=80.0)
            #             keras.layers.Activation('sigmoid')
            # 128
        ]

    @tf.function
    def call(self, inputs, **kwargs):

        inputs = tf.expand_dims(inputs, -1)

        for layer in self.conv:
            #             print(inputs.shape)
            inputs = layer(inputs)

        for layer in self.dconv:
            #             print(inputs.shape)
            inputs = layer(inputs)

        inputs = tf.squeeze(inputs, axis=-1)
        return inputs
