import json


def load_config(file):
    with open(file) as f:
        data = json.load(f)
    return data
