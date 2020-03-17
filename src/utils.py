import glob
import random
import os
import time
import yaml
from contextlib import contextmanager
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
import requests
import dropbox
from notion.client import NotionClient
from collections import OrderedDict
from easydict import EasyDict as edict
from pathlib import Path


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Timer:

    def __init__(self):
        self.processing_time = 0

    @contextmanager
    def timer(self, name):
        t0 = time.time()
        yield
        t1 = time.time()
        processing_time = t1 - t0
        self.processing_time += round(processing_time, 10)
        if self.processing_time < 60:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time:.2f} sec)')
        elif self.processing_time < 3600:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 60:.2f} min)')
        else:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 3600:.2f} hour)')

    def get_processing_time(self, type='sec'):
        if type == 'sec':
            return round(self.processing_time, 2)
        elif type == 'min':
            return round(self.processing_time / 60, 2)
        elif type == 'hour':
            return round(self.processing_time / 3600, 2)


# =============================================================================
# Data Processor
# =============================================================================
class YmlPrrocessor:

    def load(self, path):
        yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                             lambda loader,
                             node: OrderedDict(loader.construct_pairs(node)))

        with open(path, 'r') as yf:
            yaml_file = yaml.load(yf, Loader=yaml.SafeLoader)
        return edict(yaml_file)

    def save(self, path, data):
        def represent_odict(dumper, instance):
            return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())

        yaml.add_representer(OrderedDict, represent_odict)
        yaml.add_representer(edict, represent_odict)

        with open(path, 'w') as yf:
            yf.write(yaml.dump(OrderedDict(data), default_flow_style=False))


class CsvProcessor:

    def __init__(self, sep):
        self.sep = sep

    def load(self, path, sep=','):
        data = pd.read_csv(path, sep=sep)
        return data

    def save(self, path, data):
        data.to_csv(path, index=False)


class FeatherProcessor:

    def load(self, path):
        data = pd.read_feather(path)
        return data

    def save(self, path, data):
        data.to_feather(path)


class PickleProcessor:

    def load(self, path):
        data = joblib.load(path)
        return data

    def save(self, path, data):
        joblib.dump(data, path, compress=True)


class NpyProcessor:

    def load(self, path):
        data = np.load(path)
        return data

    def save(self, path, data):
        np.save(path, data)


class DataProcessor:

    def __init__(self):
        self.data_encoder = {
            '.yml': YmlPrrocessor(),
            '.csv': CsvProcessor(sep=','),
            '.tsv': CsvProcessor(sep='\t'),
            '.feather': FeatherProcessor(),
            '.pkl': PickleProcessor(),
            '.npy': NpyProcessor()
        }

    def load(self, path):
        extension = self._extract_extension(path)
        data = self.data_encoder[extension].load(path)
        return data

    def save(self, path, data):
        extension = self._extract_extension(path)
        self.data_encoder[extension].save(path, data)

    def _extract_extension(self, path):
        return os.path.splitext(path)[1]


class Yml:

    def load(self, path):
        with open(path, 'r') as yf:
            yaml_file = yaml.load(yf)
        return yaml_file

    def save(self, path, data):
        with open(path, 'w') as yf:
            yf.write(yaml.dump(data, default_flow_style=False))


# =============================================================================
# Notification
# =============================================================================
def send_line(line_token, message):
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


def send_notion(token_v2, url, name, created, model, local_cv, time_, comment):
    client = NotionClient(token_v2=token_v2)
    cv = client.get_collection_view(url)
    row = cv.collection.add_row()
    row.name = name
    row.created = created
    row.model = model.split('_')[0]
    row.local_cv = local_cv
    row.time = time_
    row.comment = comment