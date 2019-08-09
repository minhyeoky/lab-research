from base.base_layer import *

Conv2DTranspose = tf.keras.layers.Conv2DTranspose
Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
ReLU = tf.keras.layers.ReLU
Reshape = tf.keras.layers.Reshape
Concatenate = tf.keras.layers.Concatenate
LeakyReLU = tf.keras.layers.LeakyReLU
K = tf.keras.backend



class Decoder(BaseLayer):

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.conv1 = Conv2DTranspose(filters=512, kernel_size=5, strides=2, padding='same')
        self.conv2 = Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same')
        self.conv3 = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same')
        self.conv4 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')
        self.conv5 = Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding='same',
                                     activation='tanh')

        self.conv = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        self.norm = [BatchNormalization(axis=-1) for _ in range(4)]
        self.relu = ReLU()

    def call(self, inputs, training=True):
        z = inputs

        for i in range(4):
            z = self.conv[i](z)
            z = self.norm[i](z)
            z = self.relu(z)
        z = self.conv[-1](z)

        self.shape = z.get_shape().as_list()
        return z


class Injector(BaseLayer):
    def __init__(self, **kwargs):
        super(Injector, self).__init__(**kwargs)

        self.reshape = Reshape((1, 1, self.config.num_attr))
        self.tile = K.tile
        self.concat = Concatenate(axis=-1)

    def call(self, inputs, training=True):
        z, attr = inputs

        z_shape = z.get_shape().as_list()

        attr = self.reshape(attr)
        attr = self.tile(attr, n=[1, z_shape[1], z_shape[2], 1])

        z_attr = self.concat([z, attr])
        return z_attr


class Encoder(BaseLayer):

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.conv1 = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')
        self.conv2 = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')
        self.conv3 = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')
        self.conv4 = Conv2D(filters=512, kernel_size=5, strides=2, padding='same')

        self.conv = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.norm = [BatchNormalization(axis=-1) for _ in range(4)]
        self.leaky_relu = LeakyReLU(self.config.leaky_relu_alpha)

    def call(self, inputs, training=True):
        z = inputs

        for i in range(4):
            z = self.conv[i](z)
            z = self.norm[i](z)
            z = self.leaky_relu(z)

        self.shape = z.get_shape().as_list()
        return z
