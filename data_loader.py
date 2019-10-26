import json
import random
from functools import partial
from pathlib import Path
from pathlib import PosixPath
from pprint import pformat
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from utils import *
import random

import logging

logger = logging.getLogger()
logger.setLevel("DEBUG")


def read_list(data_dir, min_sec, n_max=999999):
    data_dir_a = Path(os.path.join(str(data_dir), 'train_a'))
    data_dir_b = Path(os.path.join(str(data_dir), 'train_b'))

    def _read_dir(dir: PosixPath):
        ret = []
        for each in dir.glob('*.wav'):
            file_name = str(each)
            if not DataLoader.check_sec(file_name, min_sec):
                continue
            else:
                ret.append(file_name)

            if len(ret) == n_max:
                break
        return ret

    data_a = _read_dir(data_dir_a)
    data_b = _read_dir(data_dir_b)
    return data_a, data_b


class DataLoader:

    def __init__(self, config, data_dir):
        logging.info(f'DataLoader initializing')
        self.config = load_config(config)

        self.data_dir = data_dir  # data_dir dir

        # dataset list<str>
        self.data_a = None
        self.data_b = None

        # configuration json
        self.n_max = None  # 데이터 수 제한
        self.n_fft = None  # STFT N_FFT
        self.hop_length = None  # STFT frame length
        self.window = None  # STFT window function name
        self.min_sec = None  # 최소 오디오 길이
        self.win_length = None  # stft win_length
        self.sr = None
        self.augmentation = None
        self.noise_factor = None
        self.cut_audio = None
        self.y_axis = None
        self._mel_filter = None
        self.fmax = None
        self.power = None

        self.__dict__ = {**self.__dict__,
                         **self.config}

        self._build()

    @property
    def output_shape(self):
        # stft: (1 + n_fft/2, n_frames)
        # melscale: (n_mels, n_frames)
        return self.n_mels, self.sr * self.min_sec / self.hop_length


    @property
    def n_a(self):
        return len(self.data_b)

    @property
    def n_b(self):
        return len(self.data_a)

    def _build(self):
        self.data_a, self.data_b = read_list(data_dir=self.data_dir, n_max=self.n_max, min_sec=self.min_sec)
        self._mel_filter = self.mel_filter
        self._validate_build()

        logging.info('Build Done')
        logging.info('== Dataloader paramters ==')
        logging.info(pformat(self.config))
        logging.info('== Number of dataset == ')
        logging.info(f'= HUB: {self.n_a}')
        logging.info(f'= LAB: {self.n_b}')

    def _validate_build(self):
        """할당되지 않은 변수가 있는 지 확인"""
        for key, value in self.__dict__.items():
            if value is None:
                raise ValueError(f'{key} is None')

    @staticmethod
    def check_sec(filename, min_sec):
        if not isinstance(filename, str):
            raise TypeError

        sec = librosa.get_duration(filename=filename)

        if sec < min_sec:
            return False
        else:
            return True

    @staticmethod
    def _norm(y):
        """음성 크기 (진폭) -1.0~1.0 정규화"""
        div = max(y.max(), abs(y.min()))
        y = y * (1. / div)
        return y

    def generator(self, data_type):
        """audio Generator

        :return: y, sr
        """
        if data_type == 'a':
            datas = self.data_a
        elif data_type == 'b':
            datas = self.data_b
        else:
            raise TypeError
        for each in datas:
            _y, _sr = librosa.load(each, sr=self.sr)
            _y = self._prep_audio(_y)
            yield _y

    def _inject_noise(self, y):
        raise NotImplementedError
        if self.noise_factor == 0:
            noise_factor = 0.01 * np.random.random_sample(1)
        else:
            noise_factor = self.noise_factor
        y = y + np.random.randn(y.shape[0]) * noise_factor
        return y

    def _prep_audio(self, y):
        y = self._trunc_audio(y)
        if self.augmentation:
            y = self._inject_noise(y)
        y = self._norm(y)
        y = self._stft(y)
        y = np.abs(y) ** self.power
        y = np.dot(self.mel_filter, y)
        y = np.log(y)
        return y

    def post_audio(self, y):
        y = np.array(y)
        y = np.exp(y)
        y = librosa.feature.inverse.mel_to_stft(y, sr=self.sr, n_fft=self.n_fft, power=self.power)
        y = librosa.griffinlim(y, hop_length=self.hop_length,
                               win_length=self.win_length)
        return y

    def load_infer_data(self, audio_file: PosixPath):
        """Load data for inference

        :param audio_file: (PosixPath) 대상 파일 경로
        :return: Generator inputs
        """
        raise NotImplementedError
        logging.debug(audio_file)

        _y, _sr = librosa.load(audio_file, sr=self.sr)
        _y = self._prep_audio(_y)

        # Add batch axis
        _y = np.expand_dims(_y, axis=0)

        return _y

    def _stft(self, y):
        return librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, window=self.config['window'],
                            win_length=self.win_length)

    @property
    def mel_filter(self):
        if self._mel_filter is None:
            self._mel_filter = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmax=self.fmax)
            return self._mel_filter
        else:
            return self._mel_filter

    def _trunc_audio(self, y):
        audio_len = y.shape[0]
        trunc_len = self.sr * self.min_sec
        trunc_point = random.randint(trunc_len, audio_len) - 1
        _y = y[trunc_point - (trunc_len - 1):trunc_point]

        _cut_point = random.randint(0, trunc_len - 1)

        if self.cut_audio:
            if _cut_point > trunc_len // 2:
                _y[_cut_point:] = 0
            else:
                _y[:_cut_point] = 0

        return _y

    def _pad_audio(self, y):
        raise NotImplementedError
        _y = np.zeros(shape=[self.sr * self.min_sec - 1])
        # **-1** - stft time axis shape 조절
        _y[0:y.shape[0]] = y
        return _y

    def _power_to_db(self, s):
        raise NotImplementedError
        return librosa.power_to_db(s, ref=np.max)

    def specshow(self, y, return_figure=False, x_axis='time', y_axis=None):
        """plot spectrogram

        :param y:
        :return: axis
        """
        if y_axis is None:
            y_axis = self.y_axis

        fig = plt.figure()
        y = librosa.amplitude_to_db(y, ref=np.max)
        axes = librosa.display.specshow(y, hop_length=self.config['hop_length'],
                                        fmax=self.fmax,
                                        sr=self.sr,
                                        y_axis=y_axis,
                                        x_axis=x_axis,
                                        cmap='magma')

        plt.title('Magnitude spectrogram')

        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        if return_figure:
            return fig
        plt.show()

    def random_audio(self, path=None):

        if path:
            y, sr = librosa.load(path, sr=self.sr)
        else:
            y, sr = librosa.load(random.choice(self.data_b + self.data_a), sr=self.sr)

        return y, sr
