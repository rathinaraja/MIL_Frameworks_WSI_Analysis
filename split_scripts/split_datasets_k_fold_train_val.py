import os
import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

def Balanced_K_fold_Train_Val(args):
    csv_path = args.csv_path
    k = args.k
    df = pd.read_csv(csv_path)
    X = df['slide_name']
    y = df['label']
    save_dir = args.save_dir
    dataset_name = args.dataset_name
    
    skf = StratifiedKFold(n_splits=k, random_state=args.seed, shuffle=True)
    
    for k_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        # Create a dataframe with all samples
        fold_df = df.copy()
        fold_df['split'] = 'train'
        fold_df.loc[val_index, 'split'] = 'val'
        
        # Separate into train and val
        train_df = fold_df[fold_df['split'] == 'train'][['slide_name', 'label']].reset_index(drop=True)
        val_df = fold_df[fold_df['split'] == 'val'][['slide_name', 'label']].reset_index(drop=True)
        
        # Pad to same length
        max_len = max(len(train_df), len(val_df))
        train_df = train_df.reindex(range(max_len))
        val_df = val_df.reindex(range(max_len))
        
        # Create output dataframe
        one_fold = pd.DataFrame({
            'train_slide_path': train_df['slide_name'],
            'train_label': train_df['label'],
            'val_slide_path': val_df['slide_name'],
            'val_label': val_df['label'],
            'test_slide_path': [np.nan] * max_len,
            'test_label': [np.nan] * max_len
        })
        
        os.makedirs(f'{save_dir}', exist_ok=True)
        output_path = f'{save_dir}/Total_{k}-fold_{dataset_name}_{k_idx+1}fold.csv'
        one_fold.to_csv(output_path, index=False)
        
        print(f"Fold {k_idx+1}: Train={len(train_df.dropna())}, Val={len(val_df.dropna())}, "
              f"Total={len(train_df.dropna()) + len(val_df.dropna())}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--k', type=int, default=3)
    argparser.add_argument('--save_dir', type=str, default='/dir/to/save/dataset/csvs')
    args = argparser.parse_args()
    Balanced_K_fold_Train_Val(args)