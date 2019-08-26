import tensorflow as tf
import numpy as np


class LatentDecoder():

    def __init__(self):
        pass

    def build(self, X, params):
        with tf.name_scope("latent-decoder"):
            layer1 = self._build_conv1d_transpose(X, params["latent_decoder_W1"], params["latent_decoder_b1"], 4)
            layer2 = self._build_conv1d_transpose(layer1, params["latent_decoder_W2"], params["latent_decoder_b2"], 4, bn=False)
            
            layer3 = tf.reshape(layer2, (tf.shape(X)[0], -1))

            layer4 = self._build_fc(layer3, params["disc_W1"], params["disc_b1"])
            layer5 = self._build_fc(layer4, params["disc_W2"], params["disc_b2"])
            layer6 = self._build_fc(layer5, params["disc_W3"], params["disc_b3"], bn=False)

        return layer2, layer6

    def _build_conv1d_transpose(self, X, W, b, s, actv=tf.nn.tanh, bn=True):

        if s == 4:
            output_seq = tf.shape(X)[1] * s
            output_ch = tf.shape(W)[1]
        else:
            output_seq = tf.shape(X)[1]
            output_ch = tf.shape(W)[1]

        with tf.name_scope("conv1d-transpose"):
            layer = tf.nn.conv1d_transpose(
                X,
                W,
                output_shape=(tf.shape(X)[0], output_seq, output_ch),
                strides=s,
                padding="SAME"
            ) + b

            if bn is True:
                mean, var = tf.nn.moments(layer, axes=0)
                layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)

            if actv is not None:
                layer = actv(layer)

            return layer

    def _build_fc(self, X, W, b, actv=tf.nn.tanh, bn=True):
        layer = tf.matmul(X, W) + b
        if bn is True:
            mean, var = tf.nn.moments(layer, axes=0)
            layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)

        if actv is not None:
            layer = actv(layer)

        return layer

