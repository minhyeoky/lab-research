from base.base_layer import *

Reshape = tf.keras.layers.Reshape
Concatenate = tf.keras.layers.Concatenate
K = tf.keras.backend


class Injector(BaseLayer):
    def __init__(self):
        super(Injector, self).__init__(name='Injector')

        self.reshape = Reshape((1, 1, self.config.num_attr))
        self.tile = K.tile
        self.concat = Concatenate(axis=-1)

    def call(self, inputs, training=None):
        z, attr = inputs

        z_shape = z.get_shape().as_list()

        attr = self.reshape(attr)
        attr = self.tile(attr, n=[1, z_shape[1], z_shape[2], 1])
        attr = K.cast(attr, 'float32')

        z_attr = self.concat([z, attr])
        return z_attr
