from base.base_layer import *

Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
LeakyReLU = tf.keras.layers.LeakyReLU


class Encoder(BaseLayer):

    def __init__(self):
        super(Encoder, self).__init__(name='Encoder')

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
