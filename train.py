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

logging.info(f'configuration setting from {args.config}')
DataLoader = data_loader.DataLoader
config = data_loader.load_config(args.config)
logging.info(pformat(config))

batch_size = config['batch_size']
n_max = config['n_max']
learning_rate = config['learning_rate']
epochs = config['epochs']
output_dir = args.output_dir
n_steps_per_epoch = n_max // batch_size
logging.info(f'n_steps_per_epoch: {n_steps_per_epoch}')
save_step = 100  # TODO
total_step = epochs * batch_size
output_dir = Path(args.output_dir).joinpath(Path(args.config).name.strip('.json'))
ckpt_dir = Path(f'{output_dir}/ckpt')
log_dir = f'{output_dir}/tensorboard'
checkpoint_file = os.path.join(args.data_dir, 'bert_model', 'bert_model.ckpt')
weight_g = config['weight_g']
weight_d = config['weight_d']
weight_rec = config['weight_rec']
data_type = config['data_type']
# 버퍼 > n_max라면 버퍼만 채우는 오류 수정
shuffle_buffer = batch_size * 10 if batch_size * 10 < n_max else n_max

if not output_dir.exists():
    output_dir.mkdir()
if not ckpt_dir.exists():
    ckpt_dir.mkdir()

# Data
dl: DataLoader = DataLoader(config=args.config, data=args.data_dir, n_max=n_max, test=args.infer)
dataset_audio = tf.data.Dataset.from_generator(partial(dl.generator, data=data_type),
                                               output_types=tf.float32,
                                               output_shapes=dl.stft_shape)
dataset_text = tf.data.Dataset.from_generator(partial(dl.generator, data=data_type, return_text=True),
                                              output_types=tf.int32,
                                              output_shapes=dl.text_shape)
dataset_label = tf.data.Dataset.from_generator(partial(dl.generator, data=data_type, return_label=True),
                                               output_types=tf.int32)
dataset_train = tf.data.Dataset.zip((dataset_audio, dataset_text, dataset_label))
dataset_train = dataset_train.shuffle(buffer_size=shuffle_buffer).batch(batch_size)

# Train setup


logging.info(f'start training output_dir: {output_dir}')

with open(f'{output_dir}/config.json', mode='w', encoding='utf8') as f:
    f.write(pformat(config).replace('.', '"'))

writer_train = tf.summary.create_file_writer(log_dir)


# tf.summary.trace_on(graph=True, profiler=True)


def loss_l1(true, pred):
    """Reconstruction loss for generator

    ::: 기본적으로 * 10.0
    """

    def _flatten(x):
        x = tf.reshape(x, shape=[x.shape[0], -1])
        return x

    true = _flatten(true)
    pred = _flatten(pred)

    loss = tf.reduce_mean(tf.abs(true - pred), axis=-1)
    loss = tf.reduce_mean(loss) * 10.0
    # 실수로 넣은 10.0인데 어차피 가중치를 줘야하긴 했기 때문 + 기존 실험 버리기 애매하기 때문에 둠

    return loss


def loss_d(real, fake, real_true=None):
    """Ordinary DCGAN's discriminator loss"""
    # if real_true:
    loss_real = keras.losses.binary_crossentropy(real_true, real, label_smoothing=0.05)
    # else:
    #     raise ValueError
    #     loss_real = keras.losses.binary_crossentropy(tf.ones_like(real), real, label_smoothing=0.2)
    loss_fake = keras.losses.binary_crossentropy(tf.zeros_like(fake), fake, label_smoothing=0.2)
    loss = loss_real + loss_fake
    loss = tf.reduce_mean(loss)
    return loss


def loss_g(fake):
    """Ordinary DCGAN's generator loss"""
    loss = keras.losses.binary_crossentropy(tf.ones_like(fake), fake, label_smoothing=0.2)
    loss = tf.reduce_mean(loss)
    return loss


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
    """Restore STFT data to wavefile by griffinlim algorithm"""
    x = np.array(x)
    x = librosa.griffinlim(x, hop_length=dl.hop_length, win_length=dl.win_length)
    return x


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


@tf.function
def train_step(x_train, step):
    with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
        pred = gen(x_train)
        probs_fake = dis((pred, x_train[1]))
        probs_real = dis((x_train[0], x_train[1]))

        _loss_l1 = loss_l1(x_train[0], pred) * weight_rec
        _loss_g = loss_g(probs_fake) * weight_g + _loss_l1
        _loss_d = loss_d(probs_real, probs_fake, x_train[2]) * weight_d

    grads_g = tape_g.gradient(_loss_g, gen.trainable_variables)
    grads_and_vars = zip(grads_g, gen.trainable_variables)
    if step % save_step == 0:
        _summary_grads(grads_and_vars, step, model='Generator')
    opt_g.apply_gradients(zip(grads_g, gen.trainable_variables))

    grads_d = tape_d.gradient(_loss_d, dis.trainable_variables)
    grads_and_vars = zip(grads_d, dis.trainable_variables)
    if step % save_step == 0:
        _summary_grads(grads_and_vars, step, model='Discriminator')
    opt_d.apply_gradients(zip(grads_d, dis.trainable_variables))

    return pred, (_loss_g, _loss_d, _loss_l1)


# Load model
gen = Generator(config=args.config, input_shape=dl.stft_shape, checkpoint_file=checkpoint_file)
dis = Discriminator(config=args.config, input_shape=dl.stft_shape, checkpoint_file=checkpoint_file)
opt_g = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
opt_d = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

ckpt_g = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=opt_g, net=gen)
ckpt_d = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=opt_d, net=dis)
manager_g = tf.train.CheckpointManager(ckpt_g, str(ckpt_dir) + '/g', max_to_keep=10)
manager_d = tf.train.CheckpointManager(ckpt_d, str(ckpt_dir) + '/d', max_to_keep=10)
ckpt_g.restore(manager_g.latest_checkpoint)
ckpt_d.restore(manager_d.latest_checkpoint)

############################################################
######################## Train #############################
############################################################

# https://www.tensorflow.org/guide/checkpoint
if manager_g.latest_checkpoint:
    logging.info(f'Restored from {manager_g.latest_checkpoint}')
else:
    logging.info(f'Start from scratch...')


def train():
    for epoch in range(epochs):

        start = time()
        for x_train in dataset_train:
            step = ckpt_g.step
            pred, loss = train_step(x_train, ckpt_g.step)

            if int(ckpt_g.step) % 50 == 0:
                with writer_train.as_default():
                    tf.summary.scalar(name='Loss-Generator-crossE',
                                      data=loss[0],
                                      step=step,
                                      description='Total Loss of Generator')
                    tf.summary.scalar(name='Loss-Discriminator',
                                      data=loss[1],
                                      step=step,
                                      description='Total Loss of Discriminator')
                    tf.summary.scalar(name='Loss-Generator-l1',
                                      data=loss[2],
                                      step=step,
                                      description='L1 Loss of Generator')
                    tf.summary.image(name='Spectrogram',
                                     data=tf.map_fn(_pred_to_img, pred),
                                     max_outputs=3,
                                     step=step,
                                     description='복원된 스펙트로그램')
                    tf.summary.audio(name='Restored Audio',
                                     data=tf.expand_dims(tf.map_fn(_pred_to_audio, pred), axis=-1),
                                     sample_rate=dl.sr,
                                     step=step,
                                     max_outputs=3)

            if step % save_step == 0:
                # tf.saved_model.save(gen, ckpt_dir=ckpt_dir)
                save_path_g = manager_g.save()
                save_path_d = manager_d.save()
                logging.info(f'Saved checkpoint for step {ckpt_g.step.numpy()}: {save_path_g}&{save_path_d}')
                logging.info(f'Loss {loss[0].numpy() + loss[1].numpy()}')
            ckpt_g.step.assign_add(1)
            ckpt_d.step.assign_add(1)

        end = time()
        with writer_train.as_default():

            tf.summary.scalar(name='run_time', data=(end - start) / n_steps_per_epoch, step=step,
                              description='per epoch')


if args.infer:

    logging.info('Inferencing...')
    if 'auto' in args.infer_audio:
        audio_file = Path(args.data_dir + '/soundAttGAN')
    else:
        audio_file = Path(args.infer_audio)

    if not manager_g.latest_checkpoint:
        logging.error('No model checkpoint')
        raise RuntimeError


    def _infer(audio_file):
        infer_data = dl.load_infer_data(audio_file)
        if not infer_data:
            return
        logging.info('Data Loaded')
        pred = gen(infer_data)
        logging.info('Infereced')
        img = _pred_to_img(pred[0])
        audio = _pred_to_audio(pred[0])
        prediction_dir = output_dir / 'prediction'
        if not prediction_dir.exists():
            logging.info(f'Making dir: {prediction_dir}')
            prediction_dir.mkdir()

        librosa.output.write_wav(f'{prediction_dir / audio_file.name}-prediction.wav', audio, dl.sr)
        text = dl.tokenizer.decode(list(infer_data[1][0]))
        with open(f'{prediction_dir / audio_file.name}-prediction.txt', mode='w', encoding='utf8') as f:
            f.write(''.join(text))

        logging.info(f'Inferencing Done: {prediction_dir / audio_file.name}')


    if 'auto' in args.infer_audio:
        for each in audio_file.glob('*.wav'):
            _infer(each)
    else:
        _infer(audio_file)

else:
    train()
