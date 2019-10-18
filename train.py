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
from model import Generator, Discriminator
import os
from time import time
from utils import *

keras = tf.keras
K = tf.keras.backend
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel('INFO')

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.json', type=str)
parser.add_argument('--data_dir', default='../data', type=str)
parser.add_argument('--output_dir', default='../data/experiment/output', type=str)
parser.add_argument('--infer', default=False, type=bool, required=False)
parser.add_argument('--infer_audio', default='', type=str, required=False)
args = parser.parse_args()

logging.info(f'setting configuration from {args.config}')

DataLoader = data_loader.DataLoader
config = load_config(args.config)

batch_size = config['batch_size']
n_max = config['n_max']
learning_rate = config['learning_rate']
epochs = config['epochs']
weight_cycle = config['weight_cycle']
output_dir = args.output_dir
n_steps_per_epoch = n_max // batch_size
save_step = 100  # TODO
total_step = epochs * batch_size
output_dir = Path(args.output_dir).joinpath(Path(args.config).name.strip('.json'))
ckpt_dir = Path(f'{output_dir}/ckpt')
log_dir = f'{output_dir}/tensorboard'
checkpoint_file = os.path.join(args.data_dir, 'bert_model', 'bert_model.ckpt')
shuffle_buffer_size = batch_size * 10 if batch_size * 10 < n_max else n_max  # 버퍼 > n_max라면 버퍼만 채우는 오류 수정

logging.info(f'n_steps_per_epoch: {n_steps_per_epoch}')
logging.info(f'total_step: {total_step}')
logging.info(f'output_dir: {output_dir}')
logging.info(f'ckpt_dir: {ckpt_dir}')
logging.info(f'log_dir: {log_dir}')
logging.info(f'checkpoint_file: {checkpoint_file}')
logging.info(f'shuffle_buffer_size: {shuffle_buffer_size}')

if not output_dir.exists():
    output_dir.mkdir()
if not ckpt_dir.exists():
    ckpt_dir.mkdir()

# DataLoader
dl: DataLoader = DataLoader(config=args.config, data_dir=args.data_dir)
dataset_fgn = tf.data.Dataset.from_generator(partial(dl.generator, data_type='lab'),
                                             output_types=tf.float32,
                                             output_shapes=dl.stft_shape).shuffle(buffer_size=shuffle_buffer_size)
dataset_kor = tf.data.Dataset.from_generator(partial(dl.generator, data_type='hub'),
                                             output_types=tf.float32,
                                             output_shapes=dl.stft_shape).shuffle(buffer_size=shuffle_buffer_size)

dataset_train = tf.data.Dataset.zip((dataset_fgn, dataset_kor)).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)  # x [0] => 외국인 / y [1] => 한국인

# Write config.json
with open(f'{output_dir}/config.json', mode='w', encoding='utf8') as f:
    f.write(pformat(config))

writer_train = tf.summary.create_file_writer(log_dir)


# Losses
def L_GAN(orig, pred):
    """Binary cross entropy loss"""
    loss = keras.losses.binary_crossentropy(y_true=tf.ones_like(orig), y_pred=orig)
    loss -= keras.losses.binary_crossentropy(y_true=tf.ones_like(pred), y_pred=pred)
    # Argmin log(1 - D(x, G(x, z))) -> Argmin -log(D(x, G(x, z)))
    loss = tf.reduce_mean(loss)
    return loss


def L_cycle(true, pred):
    """Reconstruction L1 loss"""

    def _flatten(x):
        x = tf.reshape(x, shape=[x.shape[0], -1])
        return x

    true = _flatten(true)
    pred = _flatten(pred)

    loss = tf.reduce_mean(tf.abs(true - pred), axis=-1)
    loss = tf.reduce_mean(loss)

    return loss


# Postprocessing
def _pred_to_img(x, infer=False, infer_audio=None):
    """Visualize STFT to spectrogram that is of decibel scale"""
    fig = dl.specshow(x, return_figure=True, mel=False)
    if infer:
        fig.savefig(str(output_dir) + f'/{infer_audio}.png')
    else:
        fig.savefig('./temp.png')
    width, height = fig.canvas.get_width_height()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = np.reshape(img, newshape=[height, width, 3])
    plt.close(fig)
    return img / 255.0


def _pred_to_audio(x):
    """Restore STFT img to wavefile using griffinlim algorithm"""
    x = np.array(x)
    x = librosa.griffinlim(x, hop_length=dl.hop_length, win_length=dl.win_length)
    return x


# tensorboard
def _summary_grads(grads_and_vars, step, model):
    """Gradient log to check whether train is working or not

    1. histogram of gradients
    2. absolute mean of gradients
    """
    # https://stackoverflow.com/questions/48709128/plot-gradients-of-individual-layers-in-tensorboard
    with writer_train.as_default():
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(name=f'{model}-{var.name}/grads/histogram', data=grad, step=step)
                tf.summary.scalar(name=f'{model}-{var.name}/grads/mean', data=tf.reduce_mean(tf.abs(grad)), step=step)


def _summary_content(content, step):
    orig, pred = content
    with writer_train.as_default():
        tf.summary.image(name='Spectrogram-orig-x',
                         data=tf.map_fn(_pred_to_img, orig[0]),
                         max_outputs=5,
                         step=step)

        tf.summary.image(name='Spectrogram-orig-y',
                         data=tf.map_fn(_pred_to_img, orig[1]),
                         max_outputs=5,
                         step=step)

        tf.summary.image(name='Spectrogram-pred-x',
                         data=tf.map_fn(_pred_to_img, pred[0]),
                         max_outputs=5,
                         step=step)

        tf.summary.image(name='Spectrogram-pred-y',
                         data=tf.map_fn(_pred_to_img, pred[1]),
                         max_outputs=5,
                         step=step)

        tf.summary.audio(name='Restored Audio-orig-x',
                         data=tf.expand_dims(tf.map_fn(_pred_to_audio, orig[0]), axis=-1),
                         sample_rate=dl.sr,
                         step=step,
                         max_outputs=5)
        tf.summary.audio(name='Restored Audio-orig-y',
                         data=tf.expand_dims(tf.map_fn(_pred_to_audio, orig[1]), axis=-1),
                         sample_rate=dl.sr,
                         step=step,
                         max_outputs=5)
        tf.summary.audio(name='Restored Audio-pred-x',
                         data=tf.expand_dims(tf.map_fn(_pred_to_audio, pred[0]), axis=-1),
                         sample_rate=dl.sr,
                         step=step,
                         max_outputs=5)
        tf.summary.audio(name='Restored Audio-pred-y',
                         data=tf.expand_dims(tf.map_fn(_pred_to_audio, pred[1]), axis=-1),
                         sample_rate=dl.sr,
                         step=step,
                         max_outputs=5)


def _summary_losses(losses, step):
    loss_cycle, loss_gan, loss = losses
    with writer_train.as_default():
        tf.summary.scalar(name='loss_cycle-x',
                          data=loss_cycle[0],
                          step=step)
        tf.summary.scalar(name='loss_cycle-y',
                          data=loss_cycle[1],
                          step=step)
        tf.summary.scalar(name='loss_gan-x',
                          data=loss_gan[0],
                          step=step)
        tf.summary.scalar(name='loss_gan-y',
                          data=loss_gan[1],
                          step=step)
        tf.summary.scalar(name='loss_cycle',
                          data=loss[0],
                          step=step)
        tf.summary.scalar(name='loss_gan',
                          data=loss[1],
                          step=step)


@tf.function
def train_step(x_train, step):
    with tf.GradientTape() as tape_G, tf.GradientTape() as tape_F, tf.GradientTape() as tape_Dx, tf.GradientTape() as tape_Dy:
        x, y = x_train
        y_generated = G(x)
        x_generated = F(y)

        probs_x = Dx(x)
        probs_y = Dy(y)
        probs_x_generated = Dx(x_generated)
        probs_y_generated = Dy(y_generated)

        _loss_cycle_x = L_cycle(x, x_generated)
        _loss_cycle_y = L_cycle(y, y_generated)

        _loss_gan_x = L_GAN(probs_x, probs_x_generated)
        _loss_gan_y = L_GAN(probs_y, probs_y_generated)

        l_cycle = (_loss_cycle_x + _loss_cycle_y) * weight_cycle
        l_gan = _loss_gan_x + _loss_gan_y

    grads_G = tape_G.gradient(l_cycle + l_gan, G.trainable_variables)
    grads_F = tape_G.gradient(l_cycle + l_gan, F.trainable_variables)
    grads_Dx = tape_G.gradient(l_gan, Dx.trainable_variables)
    grads_Dy = tape_G.gradient(l_gan, Dy.trainable_variables)

    def _grads_fn(grads, model, opt):
        if step % save_step == 0:
            grads_and_vars = zip(grads, model.trainable_variables)
            _summary_grads(grads_and_vars)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    _grads_fn(grads_G, G, opt_G)
    _grads_fn(grads_F, F, opt_F)
    _grads_fn(grads_Dx, Dx, opt_Dx)
    _grads_fn(grads_Dy, Dy, opt_Dy)

    return (x, y), (x_generated, y_generated), (_loss_cycle_x, _loss_cycle_y), \
           (_loss_gan_x, _loss_gan_y), (l_cycle, l_gan)
    # return pred, (_loss_g, _loss_d, _loss_l1)


# Load model
G = Generator(args.config)  # x -> y (외국인 -> 한국인)
F = Generator(args.config)  # y -> x (한국인 -> 외국인)
Dx = Discriminator(args.config)
Dy = Discriminator(args.config)

# Create Optimizer
Adam = keras.optimizers.Adam
opt_G = Adam(learning_rate, beta_1=0.5)
opt_F = Adam(learning_rate, beta_1=0.5)
opt_Dx = Adam(learning_rate, beta_1=0.5)
opt_Dy = Adam(learning_rate, beta_1=0.5)

# Create Checkpoint & Manager
# https://www.tensorflow.org/guide/checkpoint
Checkpoint = tf.train.Checkpoint
ckpt_G = Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=opt_G, net=G)
ckpt_F = Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=opt_F, net=F)
ckpt_Dx = Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=opt_Dx, net=Dx)
ckpt_Dy = Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=opt_Dy, net=Dy)

CheckpointManager = tf.train.CheckpointManager
manager_G = CheckpointManager(ckpt_G, f'{ckpt_dir}/G', max_to_keep=20)
manager_F = CheckpointManager(ckpt_F, f'{ckpt_dir}/F', max_to_keep=20)
manager_Dx = CheckpointManager(ckpt_Dx, f'{ckpt_dir}/Dx', max_to_keep=20)
manager_Dy = CheckpointManager(ckpt_Dy, f'{ckpt_dir}/Dy', max_to_keep=20)

# Restore Checkpoints
ckpt_G.restore(manager_G.latest_checkpoint)
ckpt_F.restore(manager_F.latest_checkpoint)
ckpt_Dx.restore(manager_Dx.latest_checkpoint)
ckpt_Dy.restore(manager_Dy.latest_checkpoint)

############################################################
######################## Train #############################
############################################################

if manager_G.latest_checkpoint and manager_F.latest_checkpoint and manager_Dx.latest_checkpoint and manager_Dy.latest_checkpoint:
    logging.info(f'Restored from {ckpt_dir}')
else:
    logging.info(f'Train from scratch...')


def train():
    for epoch in range(epochs):

        # start = time()
        for x_train in dataset_train:
            step = ckpt_G.step
            orig, pred, loss_cycle, loss_gan, loss = train_step(x_train, ckpt_G.step)
            losses = (loss_cycle, loss_gan, loss)
            content = (orig, pred)

            if int(ckpt_G.step) % 50 == 0:
                _summary_losses(losses, step)
                _summary_content(content, step)

            if step % save_step == 0:
                def _save(manager):
                    save_path = manager.save()
                    logging.info(f'saved checkpoint for step {ckpt_G.step.numpy()}: {save_path}')

                _save(manager_G)
                _save(manager_F)
                _save(manager_Dx)
                _save(manager_Dy)

            logging.info(f'Loss_cycle {loss[0].numpy()}, Loss_gan: {loss[1].numpy()}')
            ckpt_G.step.assign_add(1)
            ckpt_F.step.assign_add(1)
            ckpt_Dx.step.assign_add(1)
            ckpt_Dy.step.assign_add(1)

        # end = time()
        # with writer_train.as_default():
        #
        #     tf.summary.scalar(name='run_time', data=(end - start) / n_steps_per_epoch, step=step,
        #                       description='per epoch')


if args.infer:
    raise NotImplementedError
    logging.info('Inferencing...')
#     if 'auto' in args.infer_audio:
#         audio_file = Path(args.data_dir + '/soundAttGAN')
#     else:
#         audio_file = Path(args.infer_audio)
#
#     if not manager_g.latest_checkpoint:
#         logging.error('No model checkpoint')
#         raise RuntimeError
#
#
#     def _infer(audio_file):
#         infer_data = dl.load_infer_data(audio_file)
#         if not infer_data:
#             return
#         logging.info('Data Loaded')
#         pred = gen(infer_data)
#         logging.info('Infereced')
#         img = _pred_to_img(pred[0])
#         audio = _pred_to_audio(pred[0])
#         prediction_dir = output_dir / 'prediction'
#         if not prediction_dir.exists():
#             logging.info(f'Making dir: {prediction_dir}')
#             prediction_dir.mkdir()
#
#         librosa.output.write_wav(f'{prediction_dir / audio_file.name}-prediction.wav', audio, dl.sr)
#         text = dl.tokenizer.decode(list(infer_data[1][0]))
#         with open(f'{prediction_dir / audio_file.name}-prediction.txt', mode='w', encoding='utf8') as f:
#             f.write(''.join(text))
#
#         logging.info(f'Inferencing Done: {prediction_dir / audio_file.name}')
#
#
#     if 'auto' in args.infer_audio:
#         for each in audio_file.glob('*.wav'):
#             _infer(each)
#     else:
#         _infer(audio_file)
#
# else:
#     train()
