import numpy as np
import pandas as pd
import pathlib
import os

from functools import partial
from env import INFO_FILE_PATH


def read_list():
    file_list = []

    info_file = pd.read_excel(INFO_FILE_PATH, sheet_name="Sheet1")

    def append_to_file_list(row, file_list):
        file_name = f"./data/{int(row['fileName'])}_{int(row['suffix'])}.wav"
        if os.path.exists(file_name) is False:
            return None
        
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

