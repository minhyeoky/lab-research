from base.base_layer import *

Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
LeakyReLU = tf.keras.layers.LeakyReLU
Flatten = tf.keras.layers.Flatten


class Common(BaseLayer):

    def __init__(self):
        super(Common, self).__init__(name='Common')

        self.conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv2 = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')
        self.conv3 = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')
        self.conv4 = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')
        self.conv5 = Conv2D(filters=512, kernel_size=5, strides=2, padding='same')
        self.conv6 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')

        self.conv = [self.conv1, self.conv2, self.conv3,
                     self.conv4, self.conv5, self.conv6]
        self.norm = [BatchNormalization(axis=-1) for _ in range(6)]
        self.leaky_relu = LeakyReLU(alpha=self.config.leaky_relu_alpha)
        self.flatten = Flatten()

    def call(self, inputs, training=True):
        x = inputs

        for i in range(6):
            x = self.conv[i](x)
            x = self.norm[i](x)
            x = self.leaky_relu(x)

        x = self.flatten(x)
        self.shape = x.get_shape().as_list()
        return x
