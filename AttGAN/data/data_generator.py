import numpy as np
from pathlib import Path
from config import *
import tensorflow as tf
from sklearn.model_selection import train_test_split

att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
            'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
            'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
            'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
            'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
            'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
            'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
            'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
            'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
            'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
            'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male',
        'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
att_id = [att_dict[att] + 1 for att in atts]


def check_attribute_conflict(att_batch, att_name, att_names):
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] == 1:
            _set(att, 0, 'Bangs')
        elif att_name == 'Bangs' and att[att_id] == 1:
            _set(att, 0, 'Bald')
            _set(att, 0, 'Receding_Hairline')
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] == 1:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] == 1:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name:
                    _set(att, 0, n)
    return att_batch


def _map_img(file_path):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=(64, 64))
    # image = tf.image.per_image_standardization(image)
    image = image - 127.5
    image = image / 127.5
    return image


def _map_label(label):
    random_idx = np.random.choice(range(len(label)))
    label[random_idx] = 1 - label[random_idx]
    return label


def _img_generator(dataset):
    def generator():
        for each in dataset:
            yield each

    return generator


class dataGenerator(object):
    def __init__(self, config: BaseConfig):
        self.config = config
        self.num_samples = None
        self.steps_per_epoch = None

    def get_dataset(self):
        self.num_samples = int(np.loadtxt(self.config.label_path, max_rows=1, dtype=np.int32))
        self.steps_per_epoch = self.num_samples // self.config.batch_size

        # img load & split
        img_dir = Path(self.config.img_dir)
        img_train, img_val = train_test_split(list(img_dir.glob('*.jpg')), test_size=self.config.n_val)
        img_train = np.array(list(map(str, img_train)))
        img_val = np.array(list(map(str, img_val)))

        print(f'n_train: {len(img_train)}')
        print(f'n_val: {len(img_val)}')

        # label
        print(f'loading labels.', end='\r')
        # load labels
        labels = np.loadtxt(self.config.label_path, skiprows=2, usecols=att_id, dtype=np.int8)
        labels: np.ndarray = np.clip(labels, 0, 1).astype('int8')

        print(f'loading labels..', end='\r')
        labels_train = np.zeros(shape=[len(img_train), self.config.num_attr])
        labels_val = np.zeros(shape=[len(img_val), self.config.num_attr])
        # print(f'img shape, {img_train.shape}')
        for idx, path in enumerate(img_train):
            labels_train[idx] = np.array(labels[int(str(path).split('/')[-1].split('.')[0]) - 1])
        for idx, path in enumerate(img_val):
            labels_val[idx] = np.array(labels[int(str(path).split('/')[-1].split('.')[0]) - 1])
        labels_train = np.reshape(labels_train, (-1, self.config.num_attr))
        labels_val = np.reshape(labels_val, (-1, self.config.num_attr))
        # print(f'labels_train shape, {labels_train.shape}')

        print(f'loading labels...')
        # label divide to a, b

        labels_train_a = tf.data.Dataset.from_tensor_slices(labels_train)
        labels_val_a = tf.data.Dataset.from_tensor_slices(labels_val)
        labels_train_b = np.array(list(map(_map_label, labels_train)))
        labels_val_b = np.array(list(map(_map_label, labels_val)))
        labels_train_b = tf.data.Dataset.from_tensor_slices(labels_train_b)
        labels_val_b = tf.data.Dataset.from_tensor_slices(labels_val_b)

        # list img to dataset
        print(f'loading dataset.', end='\r')
        img_dataset_train = tf.data.Dataset.from_generator(_img_generator(img_train), output_types=tf.string)
        img_dataset_val = tf.data.Dataset.from_generator(_img_generator(img_val), output_types=tf.string)
        img_dataset_train = img_dataset_train.map(_map_img)
        img_dataset_val = img_dataset_val.map(_map_img)

        # zip train
        train = tf.data.Dataset.zip((img_dataset_train, labels_train_a, labels_train_b))
        train = train.shuffle(self.config.batch_size * 100).prefetch(self.config.batch_size * 10).batch(
            self.config.batch_size)

        # zip val
        print(f'loading dataset..')
        val = tf.data.Dataset.zip((img_dataset_val, labels_val_a, labels_val_b)).repeat()
        val = val.batch(self.config.n_val)

        return train, iter(val)

    def get_steps_per_epoch(self):
        return self.steps_per_epoch + 1
