import tensorflow as tf
from utils import *

keras = tf.keras

BatchNorm = keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
DConv = keras.layers.Conv2DTranspose
Dense = keras.layers.Dense


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


class AutoEncoder(keras.models.Model):
    def __init__(self, config, checkpoint_file, input_shape, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.config = load_config(config)
        self.audio_shape = input_shape

        self.act_fn = keras.layers.LeakyReLU()
        self.kernel_size = self.config['kernel_size']

        self.embedding_table = _get_embedding_table(checkpoint_file)

        self.conv = [
            Conv2D(filters=32, kernel_size=self.kernel_size, strides=(2, 2), padding='same', activation=None,
                   input_shape=input_shape),
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
            self.act_fn,
            #             keras.layers.Flatten(),
            #             Dense(units=output_size, activation=None),
            #             self.act_fn
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
        self.embed_hidden = Dense(units=self.hidden_size[0] * self.hidden_size[1])

    @tf.function
    def call(self, inputs, **kwargs):
        audio, text = inputs
        audio = tf.expand_dims(audio, -1)

        for layer in self.conv:
            audio = layer(audio)

        audio = self._concat_text(audio, text)
        for layer in self.dconv:
            audio = layer(audio)
        audio = tf.squeeze(audio, axis=-1)

        return audio

    @property
    def hidden_size(self):
        return int(self.audio_shape[0] / 2 ** 5), int(self.audio_shape[1] / 2 ** 5)

    def _concat_text(self, audio, text):
        text = tf.gather(self.embedding_table, text)
        text = self.embed_hidden(text)
        text = tf.reshape(text, shape=[-1, 32, *self.hidden_size])
        text = tf.transpose(text, perm=[0, 2, 3, 1])
        audio = tf.concat([audio, text], axis=-1)
        return audio
