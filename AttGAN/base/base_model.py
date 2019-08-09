import tensorflow as tf

from config import *


class BaseModel(tf.keras.models.Model):

    def __init__(self, name='BaseModel'):
        self.config = BaseConfig
        super(BaseModel, self).__init__(name=name)

        print(f'model: {name} 생성됨')
