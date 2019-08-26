import tensorflow as tf
import numpy as np

from env import ATT


class LatentEncoder():

    def __init__(self):
        pass

    def build(self, X, att, params):
        with tf.name_scope("latent-encoder"):
            layer1 = self._build_conv1d(
                tf.concat([X, tf.tile(tf.reshape(att, (-1, 1, len(ATT))), (1, tf.shape(X)[1], 1))], axis=-1),
                params["latent_encoder_W1"],
                params["latent_encoder_b1"],
                s=2
            )
            layer2 = self._build_conv1d(
                tf.concat([layer1, tf.tile(tf.reshape(att, (-1, 1, len(ATT))), (1, tf.shape(layer1)[1], 1))], axis=-1),
                params["latent_encoder_W2"],
                params["latent_encoder_b2"],
                s=2
            )
            layer3 = self._build_conv1d(
                tf.concat([layer2, tf.tile(tf.reshape(att, (-1, 1, len(ATT))), (1, tf.shape(layer2)[1], 1))], axis=-1),
                params["latent_encoder_W3"],
                params["latent_encoder_b3"],
                s=2
            )
            layer4 = self._build_conv1d(
                tf.concat([layer3, tf.tile(tf.reshape(att, (-1, 1, len(ATT))), (1, tf.shape(layer3)[1], 1))], axis=-1),
                params["latent_encoder_W4"],
                params["latent_encoder_b4"],
                s=2
            )

            sex_mean, sex_sd = self._mean_sd(
                layer4,
                params["latent_encoder_sex_mean"],
                params["latent_encoder_sex_sd"],
            )

            langNat_mean, langNat_sd = self._mean_sd(
                layer4,
                params["latent_encoder_langNat_mean"],
                params["latent_encoder_langNat_sd"],
            )

            levKor_mean, levKor_sd = self._mean_sd(
                layer4,
                params["latent_encoder_levKor_mean"],
                params["latent_encoder_levKor_sd"],
            )

        return (sex_mean, sex_sd + 1e-6), (langNat_mean, langNat_sd + 1e-6), (levKor_mean, levKor_sd + 1e-6)

    def _build_conv1d(self, X, W, b, s, actv=tf.nn.tanh, bn=True):
        with tf.name_scope("conv1d"):
            layer = tf.nn.conv1d(X, W, stride=s, padding="SAME") + b
            if bn is True:
                mean, var = tf.nn.moments(layer, axes=0)
                layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-8)

            if actv is not None:
                layer = actv(layer)

            return layer

    def _mean_sd(self, X, W_mean, W_sd):
        mean = self._build_conv1d(X, W_mean, 0, s=1, actv=None, bn=False)
        sd = self._build_conv1d(X, W_sd, 0, s=1, actv=tf.nn.softplus, bn=False)

        return mean, sd

