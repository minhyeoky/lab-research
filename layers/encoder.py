import tensorflow as tf

keras = tf.keras


class Encoder(keras.layers.Layer):

    def __init__(self, config):
        self.config = config
        super(Encoder, self).__init__(name='Encoder')

        self.conv1d_input = keras.layers.Conv1D(filters=config['filter_input'],
                                              kernel_size=config['kernel_input'],
                                              padding='same',
                                              input_shape=[None, config['y_length']])

        self.maxpool = keras.layers.MaxPool1D(padding='same',
                                              strides=2)

        # self.dense_2 = keras.layers.Dense(units=config['layer_2'])

    def call(self, inputs, training=None):
        """
        :param inputs:
        x: audio signal, (batch_size, 128*512)
        :param training: None
        :return: z
        """

        x = inputs
        x = self.conv1d_input(x)
        x = self.maxpool(x)

        return x
