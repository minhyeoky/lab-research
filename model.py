import tensorflow as tf
from utils import *

keras = tf.keras

BatchNorm = keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
DConv = keras.layers.Conv2DTranspose
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
ZeroPadding2D = keras.layers.ZeroPadding2D


def _get_embedding_table(checkpoint_file):
    ckpt_loader = tf.train.load_checkpoint(checkpoint_file)
    #     model = keras_bert.load_trained_model_from_checkpoint(config_file=config_file,
    #                                                          checkpoint_file=checkpoint_file,
    #                                                          training=False,
    #                                                          trainable=None,
    #                                                          output_layer_num=1,
    #                                                          seq_len=dl.max_len)
    #     embed_table = keras_bert.get_token_embedding(model)
    #     del(model)
    embed_table = ckpt_loader.get_tensor('bert/embeddings/word_embeddings')
    del (ckpt_loader)
    return embed_table


class Generator(keras.models.Model):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, config, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.config = load_config(config)

        self.alpha = 0.2
        self.act_fn = keras.layers.LeakyReLU(alpha=self.alpha)
        self.kernel_size = 4

        self.conv = [
            Conv2D(filters=32, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None),
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
            self.act_fn
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
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        audio = inputs
        audio = tf.expand_dims(audio, -1)

        for layer in self.conv:
            audio = layer(audio)

        for layer in self.dconv:
            audio = layer(audio)
        audio = tf.squeeze(audio, axis=-1)

        return audio


class Discriminator(keras.models.Model):

    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, config, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.config = load_config(config)

        self.kernel_size = 4
        self.alpha = 0.2
        self.act_fn = keras.layers.LeakyReLU(alpha=self.alpha)

        self.conv = [
            # (256, 128, 1) input image
            Conv2D(filters=32, kernel_size=self.kernel_size, padding='same', strides=(2, 2), activation=None),
            BatchNorm(),
            self.act_fn,
            # (128, 64, 32)
            Conv2D(filters=64, kernel_size=self.kernel_size, padding='same', strides=(2, 2), activation=None),
            BatchNorm(),
            self.act_fn,
            # (64, 32, 64)
            Conv2D(filters=128, kernel_size=self.kernel_size, padding='same', strides=(2, 2), activation=None),
            BatchNorm(),
            self.act_fn,
            # (32, 16, 128)
            ZeroPadding2D(),
            # (34, 18, 128)
            Conv2D(filters=256, kernel_size=self.kernel_size, padding='valid', strides=(1, 1), activation=None),
            BatchNorm(),
            self.act_fn,
            # (31, 15, 256)
            ZeroPadding2D(),
            # (33, 17, 256)
            Conv2D(filters=1, kernel_size=self.kernel_size, padding='valid', strides=(1, 1), activation=None),
            # (30, 14, 1)
            keras.layers.Activation('sigmoid')
        ]

    @tf.function
    def call(self, inputs, **kwargs):
        orig, tar = inputs
        orig = tf.expand_dims(orig, -1)
        tar = tf.expand_dims(tar, -1)
        x = keras.layers.concatenate([orig, tar])
        # print(x.shape)

        for layer in self.conv:
            x = layer(x)
            # print(x.shape)

        return x  # (30, 7 * max_sec, 1)


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
    https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    """

    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
