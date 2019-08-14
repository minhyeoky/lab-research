import tensorflow as tf

keras = tf.keras


class Decoder(keras.layers.Layer):

    def __init__(self, config):
        self.config = config
        super(Decoder, self).__init__(name='Decoder')

        self.upsample = keras.layers.UpSampling1D(size=2,
                                                  input_shape=[config['y_length'] // 2, config['filter_input']])
        self.conv1d = keras.layers.Conv1D(filters=1,
                                          kernel_size=config['kernel_output'],
                                          padding='same')
        # Conv1DTranspose가 없기 때문에, Upsample + Conv1D를 사용함
        # https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras

    def call(self, inputs, training=None):
        """
        :param inputs:
        x: audio signal, (batch_size, ?)
        :param training: None
        :return: z
        """

        x = inputs
        x = self.upsample(x)
        x = self.conv1d(x)

        return x
