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
from keras_bert import Tokenizer
from utils import *
import random

import logging

logger = logging.getLogger()
logger.setLevel("DEBUG")


def load_config(file):
    with open(file) as f:
        data = json.load(f)
    return data


def read_list(data='../data', hub=None, test=False):
    file_list = []

    file_path = os.path.join(data, 'soundAttGAN/koreancorpus_prep.xlsx')
    data_path = os.path.join(data, 'soundAttGAN')

    if hub:
        hub = os.path.join(data, 'KsponSpeech_01')
        logging.info(f'reading hub data from {hub}')
        p = Path(hub)
        for each in p.iterdir():
            logging.debug(each)
            for wav in each.glob('*.wav'):
                try:
                    with open(str(wav).replace('.pcm.wav', '') + '.txt.prep', mode='r', encoding='utf8') as f:
                        txt = f.readline().strip('\n')
                except FileNotFoundError:
                    continue

                if not txt:
                    continue
                file_list.append({
                    "fileName": str(wav),
                    "text": txt,
                    "label": "hub"
                })
            if test:
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
            file_name = f"{data_path}/{int(row['fileName'])}_{int(row['suffix'])}.wav"

            text = row["text"]
            if text is None or text is '' or type(text) is not str:
                logging.debug('lab text is none')
                return

            file_list.append({
                "fileName": file_name,
                "text": text,
                "label": "lab"
            })

        info_file.apply(partial(_append_to_file_list, file_list=file_list), axis=1)

    assert len(file_list) != 0

    logging.info(f'total number of data: {len(file_list)}')

    return file_list


class DataLoader:

    def __init__(self, config, data, n_max, test=False):
        logging.info(f'DataLoader initializing')
        self.config = load_config(config)

        self.data = data
        self.test = test
        self.data_lab = None
        self.data_hub = None
        self.data_both = None

        # configuration json
        self.n_max = n_max
        self.n_fft = None
        self.hop_length = None
        self.window = None
        self.n_mels = None
        self.sr_hub = None
        self.fmax = None  # 최대 주파수
        self.n_mels = None  # mel spectrogram
        self.max_sec = None  # 최대 오디오 길이
        self.win_length = None  # stft win_length
        self.top_db = None  # 진폭/ 파워 데시벨 변환시 최대 데시벨
        self.vocab = None  # 텍스트 임베딩 사전
        self.max_len = None  # 텍스트 토크나이즈시 최대 토큰의 길이

        self.__dict__ = {**self.__dict__,
                         **self.config}

        self.sr = 16000
        self.tokenizer = None

        self.build()

    # @property
    # def mel_shape(self):
    #     return self.n_mels, self.n_cnns * self.n_mels

    @property
    def stft_shape(self):
        return (self.n_fft / 2) + 1, self.sr * self.max_sec / self.hop_length

    @property
    def text_shape(self):
        return (self.max_len,)

    def build(self):
        self.data_lab = read_list(data=self.data)
        self.data_hub = read_list(data=self.data, hub=True, test=self.test)

        vocab_path = os.path.join(self.data, 'bert_model', 'vocab.txt')
        logging.info(f'Reading vocab from {vocab_path}')
        self.vocab = load_vocab(vocab_path)
        logging.info(f'The number of vocab is {len(self.vocab)}')
        self.tokenizer = Tokenizer(self.vocab, cased=True)

        logging.info('Build done')
        logging.info(pformat(self.config))

        self.data_both = self.data_lab + self.data_hub
        random.shuffle(self.data_both)

        self._validate_build()

    def _validate_build(self):
        for key, value in self.__dict__.items():
            if value is None:
                raise ValueError(f'{key} is None')
        return True

    def _check_sec(self, y):
        sec = y.shape[0] / self.sr
        if sec >= self.max_sec:
            return False
        else:
            return True

    def _norm(self, y):
        div = max(y.max(), abs(y.min()))
        y = y * (1. / div)
        return y

    def _norm_stft(self, y):
        raise NotImplementedError
        _y = self._norm(y)
        _y = librosa.amplitude_to_db(_y, ref=np.max, top_db=self.top_db)
        _y = (_y + self.top_db / 2.0) / (self.top_db / 2.0)
        return _y

    def _denorm_stft(self, y):
        _y = y * (self.top_db / 2.0) - self.top_db / 2.0
        _y = librosa.db_to_amplitude(_y)
        return _y

    def train_generator(self, data, norm=True, mel_spectrogram=False, return_text=False, return_label=False):
        """audio Generator

        :return: y, sr
        """
        if return_text == True and return_label == True:
            raise ValueError
        if data == 'lab':
            files = self.data_lab
        elif data == 'hub':
            files = self.data_hub
        elif data == 'both':
            files = self.data_both
        else:
            raise TypeError

        _n_iter = 0
        _mel_filter = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                          fmax=self.fmax)

        for each in files:
            file_name = each['fileName']
            text = each['text']
            label = each['label']
            _y, _sr = librosa.load(file_name, sr=self.sr)

            if not self._check_sec(_y):
                continue

            if return_label:
                if label == 'hub':
                    yield 1
                else:
                    yield 0
                continue
            if return_text:
                logging.debug(text)
                text = self.tokenizer.encode(text, second=None, max_len=self.max_len)[0]
                yield text
                continue

            _y = self._pad_audio(_y)

            if norm:
                _y = self._norm(_y)

            _y = self._stft(_y)
            _y = np.abs(_y)
            # _y = self._norm_stft(_y)
            # _y = self._norm(_y)

            if mel_spectrogram:
                raise NotImplementedError
                _y = np.dot(_mel_filter, _y)

            if self.n_max is not None:
                if _n_iter == self.n_max:
                    break
            _n_iter += 1

            yield _y

    def load_infer_data(self, audio_file: PosixPath):
        """Load data for inference

        :param audio_file: (PosixPath) 대상 파일 경로
        :return: Generator inputs
        """
        logging.debug(audio_file)
        text = None

        _y, _sr = librosa.load(audio_file, sr=self.sr)

        if not self._check_sec(_y):
            logging.info(f'audio length is too long: {audio_file} - {_y.shape[0]}')
            return False

        for each in self.data_both:
            if Path(each['fileName']).name == audio_file.name:
                text = each['text']
                break

        if text is None:
            logging.error(f'audio cannot be found : {audio_file}')
            return False

        _y = self._pad_audio(_y)
        _y = self._norm(_y)
        _y = self._stft(_y)
        _y = np.abs(_y)

        logging.info(f'{audio_file}: {text}')
        text = self.tokenizer.encode(text, second=None, max_len=self.max_len)[0]

        # Add batch axis & Dummy labelŒ
        _y = np.expand_dims(_y, axis=0)
        text = np.expand_dims(text, axis=0)
        label = np.array([[0]])

        return _y, text, label

    def _stft(self, y):
        return librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, window=self.config['window'],
                            win_length=self.win_length)

    def _pad_audio(self, y):
        _y = np.zeros(shape=[self.sr * self.max_sec - 1])
        # **-1** - stft time axis shape 조절
        _y[0:y.shape[0]] = y
        return _y

    def _melspectrogram(self, y):
        """Compute a mel-scaled spectrogram

        1. stft
        2. absolute
        3. amplitude to decibel
        """
        # [shape=(n_mels, t)]
        raise NotImplementedError
        return librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.config['n_mels'], n_fft=self.config['n_fft'],
                                              S=None, hop_length=self.config['hop_length'], win_length=None,
                                              window=self.config['window'], center=True, pad_mode='reflect', power=2.0)

    def _mel_to_audio(self, mel):
        raise NotImplementedError
        return librosa.feature.inverse.mel_to_audio(mel, sr=self.config['sr_hub'], n_fft=self.config['n_fft'],
                                                    hop_length=self.config['hop_length'])

    #     def istft(self, stft_matrix):
    #         return librosa.istft(stft_matrix=stft_matrix, hop_length=self.config['hop_length'])

    def _power_to_db(self, s):
        return librosa.power_to_db(s, ref=np.max)

    def specshow(self, y, mel=False, return_figure=False):
        """plot spectrogram

        :param y:
        :return: axis
        """
        fig = plt.figure()
        if mel:
            y = self._power_to_db(y, ref=np.max)
            axes = librosa.display.specshow(y, y_axis='mel', x_axis='time')
            plt.title('Mel spectrogram')
        else:
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

    def random_audio(self, path=None, mel_to_audio=False, specshow=True):

        if path:
            raise NotImplementedError
            y, sr = librosa.load(path)

        else:
            raise NotImplementedError
            y, sr = librosa.load(random.choice(self.data_hub)['fileName'], sr=self.config['sr_hub'])

        if mel_to_audio:
            y = self._melspectrogram(y)
            self.specshow(y, mel=True)
            y = self._mel_to_audio(y)
        else:
            #             self.specshow(y, mel=False)
            pass

        return y, sr

    def test_train_generator(self):
        """

        :return: audio
        """
        it = iter(self.train_generator(data='hub'))
        r_number = random.randint(0, 100)
        for _ in range(r_number):
            _y = next(it)
        return self._mel_to_audio(_y)


if __name__ == "__main__":
    dl = DataLoader()
