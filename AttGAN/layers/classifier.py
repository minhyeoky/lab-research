from base.base_layer import *

Dense = tf.keras.layers.Dense
BatchNormalization = tf.keras.layers.BatchNormalization
LeakyReLU = tf.keras.layers.LeakyReLU


class Classifier(BaseLayer):

    def __init__(self):
        super(Classifier, self).__init__(name='Classifier')

        self.dense1 = Dense(units=1024)
        self.dense2 = Dense(units=self.config.num_attr, activation='sigmoid')

        self.norm = BatchNormalization(axis=-1)
        self.leaky_relu = LeakyReLU(alpha=self.config.leaky_relu_alpha)

    def call(self, inputs, training=True):
        x = inputs

        x = self.dense1(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        x = self.dense2(x)

        self.shape = x.get_shape().as_list()
        return x
