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


def read_list(data, max_sec, hub=None, n_max=999999):
    file_list = []

    file_path = os.path.join(data, 'soundAttGAN/koreancorpus_prep.xlsx')
    data_path = os.path.join(data, 'soundAttGAN')

    if hub:
        file = os.path.join(data, 'KsponSpeech_01')
        logging.info(f'reading hub data from {file}')
        p = Path(file)
        for each in p.iterdir():
            logging.debug(each)
            for wav in each.glob('*.wav'):
                filename = str(wav)
                if not DataLoader.check_sec(filename, max_sec):
                    continue
                file_list.append({
                    "filename": filename,
                    "label": "hub"
                })
            if len(file_list) > n_max:
                break


    else:
        logging.info(f'reading lab data from {data_path} & {file_path}')

        info_file = pd.read_excel(file_path, sheet_name="Sheet1")

        def _append_to_file_list(row, file_list):
            """data_path 읽어서 file_list 에 넣음,

            :param row: 엑셀
            :param file_list:
            :return: None
            """
            filename = f"{data_path}/{int(row['fileName'])}_{int(row['suffix'])}.wav"

            if not DataLoader.check_sec(filename, max_sec):
                return
            file_list.append({
                "filename": filename,
                "label": "lab"
            })

        info_file.apply(partial(_append_to_file_list, file_list=file_list), axis=1)

    assert len(file_list) != 0

    logging.info(f'total number of data: {len(file_list)}')

    return file_list


class DataLoader:

    def __init__(self, config, data_dir):
        logging.info(f'DataLoader initializing')
        self.config = load_config(config)

        self.data = data_dir  # data_dir dir

        # dataset list<str>
        self.data_lab_train = None
        self.data_lab_valid = None
        self.data_hub_train = None
        self.data_hub_valid = None

        # configuration json
        self.n_max = None  # 데이터 수 제한
        self.n_fft = None  # STFT N_FFT
        self.hop_length = None  # STFT frame length
        self.window = None  # STFT window function name
        self.fmax = None  # 최대 주파수
        self.max_sec = None  # 최대 오디오 길이
        self.win_length = None  # stft win_length
        self.n_valid = None  # 검증셋의 갯수
        self.sr = None
        self.augmentation = None
        self.noise_factor = None

        self.__dict__ = {**self.__dict__,
                         **self.config}

        self._build()

    @property
    def stft_shape(self):
        return (self.n_fft / 2) + 1, self.sr * self.max_sec / self.hop_length

    @property
    def n_hub(self):
        return len(self.data_hub_train) + len(self.data_hub_valid)

    @property
    def n_lab(self):
        return len(self.data_lab_train) + len(self.data_lab_valid)

    def _build(self):
        data_hub = read_list(data=self.data, hub=True, n_max=self.n_max, max_sec=self.max_sec)
        data_lab = read_list(data=self.data, max_sec=self.max_sec)

        data_hub = data_hub[:self.n_max]
        data_lab = data_lab[:self.n_max]

        self.data_lab_train = data_lab[self.n_valid:]
        self.data_hub_train = data_hub[self.n_valid:]
        self.data_lab_valid = data_lab[:self.n_valid]
        self.data_hub_valid = data_hub[:self.n_valid]

        self._validate_build()

        logging.info('Build Done')
        logging.info('== Dataloader paramters ==')
        logging.info(pformat(self.config))
        logging.info('== Number of dataset == ')
        logging.info(f'= HUB: {self.n_hub}')
        logging.info(f'= LAB: {self.n_lab}')

    def _validate_build(self):
        """할당되지 않은 변수가 있는 지 확인"""
        for key, value in self.__dict__.items():
            if value is None:
                raise ValueError(f'{key} is None')

    @staticmethod
    def check_sec(filename, max_sec):
        if not isinstance(filename, str):
            raise TypeError

        sec = librosa.get_duration(filename=filename)

        if sec < max_sec:
            return False
        else:
            return True

    @staticmethod
    def _norm(y):
        """음성 크기 (진폭) -1.0~1.0 정규화"""
        div = max(y.max(), abs(y.min()))
        y = y * (1. / div)
        return y

    def generator(self, data_type, valid=False, norm=True, return_label=False):
        """audio Generator

        :return: y, sr
        """
        if data_type == 'lab':
            if valid:
                files = self.data_lab_valid
            else:
                files = self.data_lab_train
        elif data_type == 'hub':
            if valid:
                files = self.data_hub_valid
            else:
                files = self.data_hub_train
        else:
            raise TypeError

        for each in files:
            file_name = each['filename']
            _y, _sr = librosa.load(file_name, sr=self.sr)
            _y = self._prep_audio(_y)
            yield _y

    def _inject_noise(self, y):
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
        y = np.abs(y)
        return y

    def load_infer_data(self, audio_file: PosixPath):
        """Load data for inference

        :param audio_file: (PosixPath) 대상 파일 경로
        :return: Generator inputs
        """
        logging.debug(audio_file)

        _y, _sr = librosa.load(audio_file, sr=self.sr)
        _y = self._prep_audio(_y)

        # Add batch axis
        _y = np.expand_dims(_y, axis=0)

        return _y

    def _stft(self, y):
        return librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, window=self.config['window'],
                            win_length=self.win_length)

    def _trunc_audio(self, y):
        audio_len = y.shape[0]
        trunc_len = self.sr * self.max_sec
        if self.augmentation:
            trunc_point = random.randint(trunc_len, audio_len) - 1
        else:
            trunc_point = trunc_len - 1
        _y = y[trunc_point - (trunc_len - 1):trunc_point]
        return _y

    def _pad_audio(self, y):
        raise NotImplementedError
        _y = np.zeros(shape=[self.sr * self.max_sec - 1])
        # **-1** - stft time axis shape 조절
        _y[0:y.shape[0]] = y
        return _y

    def _power_to_db(self, s):
        raise NotImplementedError
        return librosa.power_to_db(s, ref=np.max)

    def specshow(self, y, mel=False, return_figure=False):
        """plot spectrogram

        :param y:
        :return: axis
        """
        fig = plt.figure()
        y = librosa.amplitude_to_db(y, ref=np.max)
        axes = librosa.display.specshow(y, hop_length=self.config['hop_length'],
                                        fmax=self.fmax,
                                        sr=self.sr,
                                        y_axis='linear',
                                        x_axis='time')
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
            y, sr = librosa.load(random.choice(self.data_hub_train + self.data_lab_train)['filename'], sr=self.sr)

        return y, sr
