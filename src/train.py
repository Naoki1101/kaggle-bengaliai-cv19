import gc
import os
import argparse
import datetime
from datetime import date
from collections import Counter, defaultdict
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import joblib

import torch

from utils import Timer, seed_torch, DataProcessor
from utils import send_line, send_notion
from dataset import BengaliDataset
from trainer import train_model, save_png

import warnings
warnings.filterwarnings('ignore')


# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument('--common', default='../configs/common/default.yml')
parser.add_argument('--notify', default='../configs/common/notify.yml')
parser.add_argument('-m', '--model')
parser.add_argument('-c', '--comment')
options = parser.parse_args()

dp = DataProcessor()
config = dp.load(options.common)
config.update(dp.load(f'../configs/exp/{options.model}.yml'))


# ===============
# Constants
# ===============
comment = options.comment
now = datetime.datetime.now()
model_name = options.model
run_name = f'{model_name}_{now:%Y%m%d%H%M%S}'

compe_params = config.compe
data_params = config.data
train_params = config.train_params
setting_params = config.settings
notify_params = dp.load(options.notify)

logger_path = Path(f'../logs/{run_name}')


# ===============
# Main
# ===============
t = Timer()
seed_torch(compe_params.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger_path.mkdir()
logging.basicConfig(filename=logger_path / 'train.log', level=logging.DEBUG)

dp.save(logger_path / 'config.yml', config)

with t.timer('load data'):
    root = Path(data_params.input_root)
    train_df = dp.load(root / data_params.train_file)

    train_img_df = pd.DataFrame()
    for file in data_params.img_file:
        df = dp.load(root / file)
        train_img_df = pd.concat([train_img_df, df], axis=0).reset_index(drop=True)

    print(train_img_df.shape)

    # img_id = train_img_df['image_id']
    # train_df = train_df[train_df['image_id'].isin(img_id)].reset_index(drop=True)

# with t.timer('add new labels'):
    # new_labels = {
    #     168: ['ত্র', 'ত্রা', 'ত্রি', 'ত্রী', 'ত্রু', 'ত্রে', 'ত্রৈ', 'ত্রো', 'ত্র্য', 'র্ত্রী', 'র্ত্রে'],
    #     169: ['স্ত্র', 'স্ত্রা', 'স্ত্রি', 'স্ত্রী', 'স্ত্রে', 'স্ত্রো'],
    #     170: ['স্র', 'স্রা', 'স্রে', 'স্রো'],
    #     171: ['ভ্র', 'ভ্রা', 'ভ্রু', 'ভ্রূ'],
    #     172: ['ষ্ক্রি', 'ষ্ক্রী'],
    #     173: ['স্ক্র', 'স্ক্রি', 'স্ক্রু']
    # }

    # new_labels = {
    #     7: ['র্দ্র', 'র্ত্রে', 'র্ত্রী']
    # }

    # for new_label, graphemes in new_labels.items():
    #     idx = train_df[train_df['grapheme'].isin(graphemes)].index
    #     train_df.loc[idx, 'consonant_diacritic'] = new_label

# with t.timer('replace new labels'):
#     new_label_df = dp.load(root / 'train_multi_diacritics.csv')
#     new_label_id = new_label_df['image_id'].values
#     replace_idx = train_df[train_df['image_id'].isin(new_label_id)].index
#     train_df.loc[replace_idx, 'consonant_diacritic'] = 7

with t.timer('sampling'):
    if setting_params.sampling_num != -1:
        use_idx = []
        for g in train_df['grapheme'].unique():
            idx_arr = np.random.choice(train_df[train_df['grapheme'] == g].index, setting_params.sampling_num)
            for idx in idx_arr:
                use_idx.append(idx)
    else:
        use_idx = list(train_df.index)

    train_df = train_df.iloc[use_idx].reset_index(drop=True)
    train_img_df = train_img_df.iloc[use_idx].reset_index(drop=True)
    train_df = pd.merge(train_df, train_img_df, on='image_id', how='left')
    del train_img_df; gc.collect()

with t.timer('create weights'):
    weights = {}
    for col in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
        weights[col] = train_df[col].value_counts().sort_index().values.astype(np.float32)

with t.timer('drop several rows'):
    if data_params.drop_fname is not None:
        drop_idx = dp.load(f'../pickle/{data_params.drop_fname}.npy')
        train_df = train_df.drop(drop_idx, axis=0).reset_index(drop=True)

with t.timer('make folds'):
    folds = pd.DataFrame(index=train_df.index)
    folds['fold_id'] = 0
    kf = KFold(n_splits=20, shuffle=True, random_state=compe_params.seed)
    for fold_, (trn_idx, val_idx) in enumerate(kf.split(train_df)):
        folds.loc[val_idx, 'fold_id'] = fold_
    folds['fold_id'] = folds['fold_id'].astype(int)

with t.timer('train model'):
    fold_num = data_params.fold_num
    x_trn = train_df[folds['fold_id'] != fold_num]
    x_val = train_df[folds['fold_id'] == fold_num]

    # pretrained: False
    num_classes = train_params.model_params.n_classes
    model_wight, oof_list, best_score, train_loss_list, val_loss_list, val_score_list = train_model(x_trn, 
                                                                                                    x_val, 
                                                                                                    train_params,
                                                                                                    num_classes,
                                                                                                    weights,
                                                                                                    device)
    np.save(f'../logs/{run_name}/oof_gr.npy', oof_list[0])
    np.save(f'../logs/{run_name}/oof_vo.npy', oof_list[1])
    np.save(f'../logs/{run_name}/oof_co.npy', oof_list[2])

    torch.save(model_wight, f'../logs/{run_name}/weight_best.pt')
    save_png(run_name, train_params, train_loss_list, val_loss_list, val_score_list)

logging.disable(logging.FATAL)
logger_path.rename(f'../logs/{run_name}_{best_score:.3f}')

process_minutes = t.get_processing_time(type='hour')

with t.timer('notify'):
    message = f'''{model_name}\ncv: {best_score:.3f}\ntime: {process_minutes:.2f}[h]'''
    send_line(notify_params.line.token, message)

    send_notion(token_v2=notify_params.notion.token_v2,
                url=notify_params.notion.url,
                name=run_name,
                created=now,
                model=train_params.model_params.model_name,
                local_cv=round(best_score, 4),
                time_=process_minutes,
                comment=comment)
