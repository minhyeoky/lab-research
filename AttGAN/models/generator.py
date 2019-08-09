from base.base_model import *
from layers.Generator.encoder import *
from layers.Generator.decoder import *
from layers.Generator.injector import *
from layers.Dicriminator_Classifier.discriminator import *

binary_crossentropy = tf.keras.losses.binary_crossentropy
mean_squared_error = tf.keras.losses.mean_squared_error
mean_absolute_error = tf.keras.losses.mean_absolute_error
K = tf.keras.backend


class Generator(BaseModel):

    def __init__(self):
        super(Generator, self).__init__(name='Generator')

        self.encoder = Genc()
        self.injector = Injector()
        self.decoder = Gdec()

    def call(self, inputs, training=True):
        xa, a, b, model = inputs
        z = self.encoder(xa)

        if not training:

            # xa_
            za = self.injector((z, a))
            xa_ = self.decoder(za)

            # xb_
            zb = self.injector((z, b))
            xb_ = self.decoder(zb)

            self.trainable = False
            return xa_, xb_

        if training:

            c_xa, c_xb_, d_xa, d_xb_ = model((xa, a, b, self), training=False)
            # L_adv
            loss_adv = mean_squared_error(K.flatten(d_xb_), K.flatten(K.ones_like(d_xb_)))
            loss_adv = loss_adv
            self.add_loss(loss_adv)

            # L_cls
            loss_cls = binary_crossentropy(K.flatten(b), K.flatten(c_xb_))
            loss_cls = loss_cls * self.config.lambda_L_cls_g
            self.add_loss(loss_cls)

            # xa_
            za = self.injector((z, a))
            xa_ = self.decoder(za)

            # L_rec
            loss_rec = mean_absolute_error(K.flatten(xa), K.flatten(xa_))
            loss_rec = loss_rec * self.config.lambda_L_rec
            self.add_loss(loss_rec)

            self.trainable = True
            return loss_adv + loss_cls + loss_rec
