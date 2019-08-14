import numpy as np
import pandas as pd
import pathlib
import os

from functools import partial
from env_mhlee import *


def read_list():
    file_list = []

    info_file = pd.read_excel(INFO_FILE_PATH, sheet_name="Sheet1")

    def append_to_file_list(row, file_list):
        """row 읽어서 file_list 에 넣음,

        :param row: 엑셀
        :param file_list:
        :return: None
        """
        file_name = f"{DATA_PATH}/{int(row['fileName'])}_{int(row['suffix'])}.wav"
        # ./data -> DATAPATH
        text = row["text"]
        sex = row["sex"]
        langNat = row["langNat"]
        levKor = row["levKor"]

        file_list.append({
                "fileName": file_name,
                "text": text,
                "sex": sex.replace("남", "M").replace("여", "F"),
                "langNat": langNat,
                "levKor": levKor
        })

        return None
        
    info_file.apply(partial(append_to_file_list, file_list=file_list), axis=1)

    return file_list

