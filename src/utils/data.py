import os
import pickle
import json


def load_data(dir_path: str):
    binary = os.path.join(dir_path)
    with open(binary, 'rb') as c:
        data = pickle.load(c)

    print(f'loaded {dir_path} to disk')
    return data

def load_json(dir_path: str):
    with open(dir_path) as f:
        data = json.load(f)

    return data