import tensorflow as tf
from config import *


class BaseLayer(tf.keras.layers.Layer):

    def __init__(self, name='BaseLayer'):
        self.shape = None
        self.config = BaseConfig
        super(BaseLayer, self).__init__(name=name)

        print(f'layer: {name} 생성됨')

    def get_shape(self):
        return self.shape
