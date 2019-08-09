from base.base_layer import *

Conv2DTranspose = tf.keras.layers.Conv2DTranspose
BatchNormalization = tf.keras.layers.BatchNormalization
ReLU = tf.keras.layers.ReLU


class Decoder(BaseLayer):

    def __init__(self):
        super(Decoder, self).__init__(name='Decoder')

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
