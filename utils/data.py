import os
import pickle


def load_data(dir_path: str):
    binary = os.path.join(dir_path)
    with open(binary, 'rb') as c:
        data = pickle.load(c)

    print(f'loaded {dir_path} to disk')
    return data

