from base.base_model import *

from layers.classifier import *
from layers.discriminator import *
from layers.common import *

binary_crossentropy = tf.keras.losses.binary_crossentropy
mean_squared_error = tf.keras.losses.mean_squared_error
K = tf.keras.backend


class DC(BaseModel):

    def __init__(self, **kwargs):
        super(DC, self).__init__(**kwargs)

        self.common = Common()
        self.discriminator = Discriminator()
        self.classifier = Classifier()

    def call(self, inputs, training=None, mask=None):
        """
        Discriminator & Classifier of AttGAN
        inputs: (xa, xb_)

        :return: probs of discriminator and classifier
        """
        xa, xb_ = inputs

        za = self.common(xa)
        zb = self.common(xb_)

        c_xa = self.classifier(za)
        c_xb_ = self.classifier(zb)

        d_xa = self.discriminator(za)
        d_xb_ = self.discriminator(zb)

        return (d_xa, d_xb_), (c_xa, c_xb_)
