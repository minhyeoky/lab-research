import numpy as np
import librosa

from preprocess import read_list
# from env import *
from env_mhlee import *


class DataLoader():

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data_list = None

    def build(self):
        """엑셀파일 읽고 배치의 수, 데이터의 수"""
        self.data_list = read_list()
        self.n = len(self.data_list)
        self.num_batches = int(np.ceil(self.n / (self.batch_size // 4)))

    def next_batch(self):
        """Batch Generator

        1. shuffle
        2. 128 * 512 만큼 cropping

        :return:
        1. y_batch: (batch_size, 128*512)
        2. att_batch: (batch_size, 3)
        """

        np.random.shuffle(self.data_list)

        for b in range(self.num_batches):
            start = b * (self.batch_size // 4)
            end = min(self.n, (b + 1) * (self.batch_size // 4))

            y_batch = np.zeros(((end - start) * 4, 128 * 512))
            att_batch = np.zeros(((end - start) * 4, NUM_ATT))

            for i in range(start, end):
                data = self.data_list[i]
                file_name = data["fileName"]
                attributes = [
                    SEX_ONE_HOT[data["sex"]],
                    langNat_ONE_HOT[data["langNat"]],
                    levKor_ONE_HOT[data["levKor"]]
                ]

                y = self._read_one_file(file_name)

                # print(len(y))

                for j in range(4):
                    y_cropped = self._crop(y, 128 * 512)

                    y_batch[(i - start) * 4 + j] = y_cropped
                    att_batch[(i - start) * 4 + j] = attributes

            yield y_batch, att_batch

    def _read_one_file(self, path):
        y, sr = librosa.load(path)
        # y = librosa.resample(y, sr, 4410)

        return y

    def _crop(self, y, length):
        """크롭핑
        음성이 Length보다 짧으면 0으로패딩
        길면 랜덤한 위치 크롭핑

        :return: ndarray, (length,)
        """
        res = np.zeros((length,))

        n = len(y)
        if n <= length:
            res[:len(y)] = y
        else:
            r = np.random.randint(0, n - length)
            res = y[r: r + length]

        return res
