import numpy as np
import pandas as pd


def main():
    train_df = pd.read_csv('../data/input/train.csv')

    # 01.特定の文字を除去
    drop_grapheme = [
        'ণ্ঠি ', 'ন্ঠি', 'ত্নী', 'ন্ঠে', 'ণ্ঠ', 'ণ্টু', 'ত্রু', 'ন্ঠ', 'ন্ডে', 'ণ্ডা', 'ণ্ডে', 'ণ্ডি', 
        'ণ্ডী', 'ণ্ড', 'হ্মা', 'ন্ট', 'ণ্টা', 'ট্ট', 'ণ্ট', 'ন্টা', 'ট্টো', 'হ্মী'
    ]

    drop_idx = train_df[train_df['grapheme'].isin(drop_grapheme)].index.values
    np.save('../pickle/drop_idx01.npy', drop_idx)


    # 02.oofを元に除去（上位5件に入っていないもの）
    oof0 = np.load('../pickle/oof0_gr.npy')
    oof1 = np.load('../pickle/oof1_gr.npy')
    oof2 = np.load('../pickle/oof2_gr.npy')
    oof3 = np.load('../pickle/oof3_gr.npy')
    oof = np.vstack((oof0, oof1, oof2, oof3))
    pred_labels = np.argsort(oof, axis=1)[:, ::-1]

    bad_sample = []
    for idx in train_df.index:
        true_label = train_df.loc[idx, 'grapheme_root']
        image_id = train_df.loc[idx, 'image_id']
        order = np.where(pred_labels[idx, :] == true_label)[0][0]
        if order > 5:
            bad_sample.append(image_id)
    
    drop_idx = train_df[train_df['image_id'].isin(bad_sample)].index.values
    np.save('../pickle/drop_idx02.npy', drop_idx)


if __name__ == '__main__':
    main()
