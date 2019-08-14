import numpy as np
import librosa

from preprocess import read_list
from env import *


class DataLoader():

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data_list = None


    def build(self):
        self.data_list = read_list()
        self.n = len(self.data_list)
        self.num_batches = int(np.ceil(self.n / (self.batch_size//4)))

    def next_batch(self):
        
        np.random.shuffle(self.data_list)

        for b in range(self.num_batches):
            start = b * (self.batch_size//4)
            end = min(self.n, (b+1)*(self.batch_size//4))

            y_batch = np.zeros(((end - start)*4, 128*512))
            att_batch = np.zeros(((end - start)*4, NUM_ATT))

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
                    y_cropped = self._crop(y, 128*512)

                    y_batch[(i - start)*4 + j] = y_cropped
                    att_batch[(i - start)*4 + j] = attributes

            yield y_batch, att_batch

    def _read_one_file(self, path):
        y, sr = librosa.load(path)
        # y = librosa.resample(y, sr, 4410)

        return y

    def _crop(self, y, length):
        res = np.zeros((length,))
        
        n = len(y)
        if n <= length:
            res[:len(y)] = y
        else:
            r = np.random.randint(0, n - length)
            res = y[r : r + length]

        return res
