import os
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

base_dir = os.path.dirname(os.path.abspath(__file__))


class BaseConfig(object):

    leaky_relu_alpha = 0.3
    num_attr = 13

    # model parameter
    lambda_L_rec = 100.0
    lambda_L_cls_g = 1.0
    lambda_L_cls_c = 1.0
    learning_rate = 0.0001

    img_dir = '../data/CelebA/img_align_celeba'
    label_path = '../data/CelebA/Anno/list_attr_celeba.txt'
    batch_size = 128
    epochs = 200

    # callbacks
    log_dir = '../data/output_dir'
    filepath = '../data/output_dir/models_001'
    update_freq = 10
    verbose = 1

    n_val = 1

    pass


class DockerConfig(BaseConfig):
    img_dir = '/tf/data/CelebA/img_align_celeba'
    label_path = '/tf/data/CelebA/Anno/list_attr_celeba.txt'
    log_dir = '/tf/data/output_dir'
    filepath = '/tf/data/output_dir/models_001'


class DevConfig(BaseConfig):
    pass
