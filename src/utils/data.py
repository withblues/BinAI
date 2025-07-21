import os
import pickle
import json
import pandas as pd
from datasets import load_from_disk

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

def load_df_and_get_model_name(filepath):
    df = pd.read_csv(filepath)
    model_name = os.path.basename(filepath).split("-results")[0]

    print(f'loaded dataframe {filepath}')
    return df, model_name

def load_dataset_and_get_model_name(filepath):
    dataset = load_from_disk(filepath)
    model_name = os.path.basename(filepath).split("-test")[0]

    print(f'loaded dataset {filepath}')
    return dataset, model_name

def load_json_and_get_model_name(filepath):
    with open(filepath) as f:
        data = json.load(f)

    model_name = os.path.basename(filepath).split("-metadata")[0]

    print(f'loaded dataset {filepath}')
    return data, model_name