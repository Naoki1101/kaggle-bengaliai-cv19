import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.formula.api import ols
from pathlib import Path
from easydict import EasyDict as edict
from notion.client import NotionClient

from utils import DataHandler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

dh = DataHandler()


class History:

    def __init__(self, log_fname=None):
        self.log_fname = log_fname
        if self.log_fname is not None:
            self.root = Path(f'../logs/{log_fname}')
            self.config = edict(dh.load(self.root / 'config.yml'))
            self.features = edict(dh.load(self.root / 'features.yml'))

    def set_fname(self, log_fname):
        self.log_fname = log_fname
        self.root = Path(f'../logs/{self.log_fname}')
        self.config = edict(dh.load(self.root / 'config.yml'))
        self.features = edict(dh.load(self.root / 'features.yml'))

    def set_latest_fname(self):
        fnames = os.listdir('../logs')
        fnames_dict = {}
        for fname in fnames:
            if len(fname.split('_')) == 3:
                fnames_dict[fname] = fname.split('_')[1]

        latest_fname = sorted(fnames_dict.items(), key=lambda x: x[1])[-1][0]
        if self.log_fname is None:
            self.set_fname(latest_fname)
        print(latest_fname)

    def get_config(self):
        return self.config

    def get_features(self):
        return self.features

    def show_importances(self):
        importance_df = dh.load(self.root / 'importance.csv')

        plt.figure(figsize=(14, importance_df['feature'].nunique() / 2))
        sns.barplot(x='importance',
                    y='feature',
                    data=importance_df.sort_values(by='importance',
                                                   ascending=False))
        plt.title(f'{self.log_fname} Features (avg over folds)')
        plt.tight_layout()
        plt.show()

    def show_oof_distribution(self):
        oof = dh.load(self.root / 'oof.npy')
        target_df = dh.load(f'../features/{self.config.train_params.data.target_name}.feather')
        target_df['oof'] = oof
        metrics = self.config.train_params.metrics

        plt.figure(figsize=(16, 8))
        if metrics in ['mae', 'rmse', 'rmpe', 'rmsle']:
            sns.jointplot(x='oof', y=self.config.train_params.data.target_name, data=target_df)

        elif metrics == 'auc':
            sns.distplot(oof[target_df[self.config.train_params.data.target_name == 0]], label='target=0')
            sns.distplot(oof[target_df[self.config.train_params.data.target_name == 1]], label='target=1')
            plt.legend()
            plt.xlim([-0.05, 1.05])
            plt.xlabel('probability')
            plt.title('distribution')

        else:
            print('Coming soon ...')

        plt.tight_layout()
        plt.show()

    def show_submit_summary(self, highlight_num=5):
        notify_params = edict(dh.load('../configs/common/notify.yml'))

        summary_df = self._get_table(notify_params.notion)
        corr = np.corrcoef(summary_df['cv_score'], summary_df['public_score'])[0, 1]
        min_ = min(np.percentile(summary_df['cv_score'], 5),
                   np.percentile(summary_df['public_score'], 5)) * 0.99
        max_ = max(np.percentile(summary_df['cv_score'], 99),
                   np.percentile(summary_df['public_score'], 99)) * 1.025

        plt.figure(figsize=(8, 8))
        plt.plot([min_, max_], [min_, max_], 'gray', ls='--', alpha=0.1)
        for i, v in summary_df.iterrows():
            if i == 0:
                s, c, marker, alpha, ec = max(100, 10 * len(summary_df)), 'r', '*', 0.7, 'y'
            elif i <= highlight_num - 1:
                s, c, marker, alpha, ec = 30 + 10 * (highlight_num - i), 'r', 'o', 0.5, 'y'
            else:
                s, c, marker, alpha, ec = 30, 'b', 'o', 0.1, None

            plt.scatter(v['cv_score'], v['public_score'], s=s, marker=marker, c=c, alpha=alpha, edgecolors=ec)

        plt.xlim([min_, max_])
        plt.ylim([min_, max_])
        plt.xlabel('cv_score')
        plt.ylabel('public_score')
        plt.title(f'CV - Public (corr: {corr:.4f})')
        plt.show()

    # def show_learning_curve(self, agg=np.mean, interval=1):
    #     log_dict = edict(dh.load(f'../logs/{self.log_fname}/train_log.json'))
    #     train_score_arr, val_score_arr, kaggle_score_arr = self._get_log_scores(log_dict)

    #     use_idx = []
    #     for i in range(len(train_score_arr)):
    #         if (i + 1) % interval == 0:
    #             use_idx.append(i)

    #     train_scores = agg(train_score_arr, axis=1)[use_idx]
    #     val_scores = agg(val_score_arr, axis=1)[use_idx]
    #     kaggle_scores = agg(kaggle_score_arr, axis=1)[use_idx]

    #     if np.sum(kaggle_scores) != 0:
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    #         ax1.plot(train_scores, 'o-', c='b', label='train_score')
    #         ax1.plot(val_scores, 'o-', c='orange', label='val_score')
    #         ax1.legend()
    #         ax1.set_xlabel('iteration')
    #         ax1.set_ylabel('score')

    #         ax2.plot(kaggle_score_arr, 'o-', c='b', label='kaggle_score')
    #         ax2.set_xlabel('epoch')
    #         ax2.set_ylabel('kaggle score')

    #     else:
    #         plt.figure(figsize=(12, 5))
    #         plt.plot(train_scores, 'o-', c='b', label='train_score')
    #         plt.plot(val_scores, 'o-', c='orange', label='val_score')
    #         plt.legend()
    #         plt.xlabel('epoch')
    #         plt.ylabel('score')

    #     plt.tight_layout()
    #     plt.show()

    def _get_table(self, params):
        client = NotionClient(token_v2=params.token_v2)
        cv = client.get_collection_view(params.url)
        rows = cv.default_query().execute()

        cv_id, public_id = self._get_columns_id(rows[0])

        datetime_list, cv_score_list, public_score_list = [], [], []
        for row in rows:
            if 'properties' in row.get():
                v = row.get()['properties']
                if public_id in v:
                    start_date = re.findall(r"'start_date': '(.+)',", str(v))[0]
                    start_time = re.findall(r"'start_time': '(.+)'}", str(v))[0]
                    t = datetime.strptime(start_date + ' ' + start_time, '%Y-%m-%d %H:%M')
                    cv_score = float(v[cv_id][0][0])
                    public_score = float(v[public_id][0][0])

                    datetime_list.append(t)
                    cv_score_list.append(cv_score)
                    public_score_list.append(public_score)

        df = pd.DataFrame({
            'datetime': datetime_list,
            'cv_score': cv_score_list,
            'public_score': public_score_list
        })
        df.sort_values(by='datetime')
        return df

    def _get_columns_id(self, row):
        for c in row.schema:
            if c['name'] == 'local_cv':
                cv_score_id = c['id']
            elif c['name'] == 'public_score':
                public_score_id = c['id']
        return cv_score_id, public_score_id

    def _get_log_scores(self, log_dict):
        fold_num = len(log_dict.iterations)
        iterations = len(log_dict.iterations.fold0)

        train_score_arr = np.zeros((iterations, fold_num))
        val_score_arr = np.zeros((iterations, fold_num))
        kaggle_score_arr = np.zeros((iterations, fold_num))

        for c, (_, fold_log) in enumerate(log_dict.iterations.items()):
            for log in fold_log:
                idx = int(log['iteration'])
                train_score_arr[idx, c] += log['train_score']
                val_score_arr[idx, c] += log['val_score']
                if 'kaggle_score' in log:
                    kaggle_score_arr[idx, c] += log['kaggle_score']
        return train_score_arr, val_score_arr, kaggle_score_arr
