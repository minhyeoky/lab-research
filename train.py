import tensorflow as tf
import librosa
import data_loader
import matplotlib.pyplot as plt
from pprint import pformat
import logging
import numpy as np
from functools import partial
from pathlib import Path
import argparse
from model import AutoEncoder
import os
from time import time

keras = tf.keras
K = tf.keras.backend
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel('INFO')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.json', type=str)
parser.add_argument('--data_dir', default='../data', type=str)
parser.add_argument('--output_dir', default='../data/experiment/output', type=str)
args = parser.parse_args()

logging.info(f'configuration setting from {args.config}')
DataLoader = data_loader.DataLoader
config = data_loader.load_config(args.config)
logging.info(pformat(config))

logger = data_loader.logger

batch_size = config['batch_size']
output_size = config['output_size']
n_max = config['n_max']
learning_rate = config['learning_rate']
epochs = config['epochs']
output_dir = args.output_dir
save_step = 50  # TODO
total_step = epochs * batch_size
output_dir = Path(args.output_dir).joinpath(Path(args.config).name.strip('.json'))
ckpt_dir = Path(f'{output_dir}/ckpt')
log_dir = f'{output_dir}/tensorboard'

if not output_dir.exists():
    output_dir.mkdir()
if not ckpt_dir.exists():
    ckpt_dir.mkdir()

# Data
dl: DataLoader = DataLoader(config=args.config, data=args.data_dir, n_max=n_max)
dataset_train = tf.data.Dataset.from_generator(partial(dl.train_generator, data='hub'),
                                               output_types=tf.float32,
                                               output_shapes=dl.stft_shape)
dataset_train = dataset_train.shuffle(buffer_size=batch_size * 10).batch(batch_size)

# Train setup


logging.info(f'start training output_dir: {output_dir}')

with open(f'{output_dir}/config.json', mode='w', encoding='utf8') as f:
    f.write(pformat(config).replace('.', '"'))

writer_train = tf.summary.create_file_writer(log_dir)


# tf.summary.trace_on(graph=True, profiler=True)


def _flatten(x):
    x = tf.reshape(x, shape=[x.shape[0], -1])
    return x


def loss_l1(x_train, pred):
    x_train = _flatten(x_train)
    pred = _flatten(pred)

    loss = tf.reduce_mean(tf.abs(x_train - pred), axis=-1)
    loss = tf.reduce_mean(loss)

    return loss


def _pred_to_img(x):
    fig = dl.specshow(x, return_figure=True, mel=False)
    fig.savefig('./temp.png')
    width, height = fig.canvas.get_width_height()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = np.reshape(img, newshape=[height, width, 3])
    plt.close(fig)
    return img / 255.0


def _pred_to_audio(x):
    x = np.array(x)
    x = librosa.griffinlim(x, hop_length=dl.hop_length, win_length=dl.win_length)
    return x


def _summary_grads(grads_and_vars, step):
    # https://stackoverflow.com/questions/48709128/plot-gradients-of-individual-layers-in-tensorboard
    with writer_train.as_default():
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(name=f'{var.name}/grads/histogram', data=grad, step=step)
                tf.summary.scalar(name=f'{var.name}/grads/mean', data=tf.reduce_mean(grad), step=step)


@tf.function
def train_step(x_train, step):
    with tf.GradientTape(persistent=True) as tape:
        pred = model(x_train)
        loss = loss_l1(x_train, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = zip(grads, model.trainable_variables)
    _summary_grads(grads_and_vars, step)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return pred, loss


# Train
# step = tf.constant(1, dtype=tf.int64)

# Load model
model = AutoEncoder(config=args.config, input_shape=dl.stft_shape)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# https://www.tensorflow.org/guide/checkpoint
ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=10)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f'Restored from {manager.latest_checkpoint}')
else:
    print(f'Start from scratch')
for epoch in range(epochs):

    start = time()
    for x_train in dataset_train:
        step = ckpt.step
        pred, loss = train_step(x_train, ckpt.step)

        if int(ckpt.step) % 50 == 0:
            with writer_train.as_default():
                tf.summary.scalar(name='loss',
                                  data=loss,
                                  step=step,
                                  description='loss 테스트')
                tf.summary.image(name='spectrogram',
                                 data=tf.map_fn(_pred_to_img, pred),
                                 max_outputs=3,
                                 step=step,
                                 description='image 테스트')
                tf.summary.audio(name='audio test',
                                 data=tf.expand_dims(tf.map_fn(_pred_to_audio, pred), axis=-1),
                                 sample_rate=dl.sr,
                                 step=step,
                                 max_outputs=3)

        if step % save_step == 0:
            # tf.saved_model.save(model, ckpt_dir=ckpt_dir)
            save_path = manager.save()
            print(f'Saved checkpoint for step {ckpt.step.numpy()}: {save_path}')
            print(f'Loss {loss.numpy()}')
        ckpt.step.assign_add(1)

    end = time()
    with writer_train.as_default():

        tf.summary.scalar(name='run_time', data=(end - start) / batch_size, step=step, description='per epoch')