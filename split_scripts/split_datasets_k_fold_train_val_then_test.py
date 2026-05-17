import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import os

def Balanced_k_fold_train_val_then_test(args):
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    test_ratio = args.test_ratio
    save_dir = args.save_dir
    dataset_name = args.dataset_name
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}\n")
    
    # First split: separate test set
    train_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        stratify=df['label'], 
        random_state=args.seed, 
        shuffle=True
    )
    
    print(f"Train+Val samples: {len(train_df)}, Test samples: {len(test_df)}\n")
    
    K = args.k
    skf = StratifiedKFold(n_splits=K, random_state=args.seed, shuffle=True)
    
    for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df['label'])):
        train_fold_df = train_df.iloc[train_index].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_index].reset_index(drop=True)
        test_fold_df = test_df.reset_index(drop=True)
        
        # Pad to same length
        max_len = max(len(train_fold_df), len(val_fold_df), len(test_fold_df))
        train_fold_df = train_fold_df.reindex(range(max_len))
        val_fold_df = val_fold_df.reindex(range(max_len))
        test_fold_df = test_fold_df.reindex(range(max_len))
        
        # Create combined dataframe
        combined_df = pd.DataFrame({
            'train_slide_path': train_fold_df['slide_path'],
            'train_label': train_fold_df['label'],
            'val_slide_path': val_fold_df['slide_path'],
            'val_label': val_fold_df['label'],
            'test_slide_path': test_fold_df['slide_path'],
            'test_label': test_fold_df['label']
        })
        
        os.makedirs(f'{save_dir}/{dataset_name}', exist_ok=True)
        output_path = f'{save_dir}/{dataset_name}/Total_{K}-fold_{dataset_name}_{fold+1}fold.csv'
        combined_df.to_csv(output_path, index=False)
        
        # Verification
        n_train = train_fold_df['slide_path'].notna().sum()
        n_val = val_fold_df['slide_path'].notna().sum()
        n_test = test_fold_df['slide_path'].notna().sum()
        n_total = n_train + n_val + n_test
        
        print(f"Fold {fold+1}: Train={n_train}, Val={n_val}, Test={n_test}, Total={n_total}")
    
    print(f"\n✓ Generated {K} folds in {save_dir}/{dataset_name}/")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--test_ratio', type=float, default=0.2)  # first select test_ratio of data as test data
    argparser.add_argument('--k', type=int, default=5)  # then split the rest of data into k folds
    argparser.add_argument('--save_dir', type=str, default='/dir/to/save/dataset/csvs')
    args = argparser.parse_args()
    Balanced_k_fold_train_val_then_test(args)
    
# ```

# **Key improvements:**

# 1. **Proper indexing**: Reset index after each split
# 2. **Clean padding**: Use `.reindex()` for consistent NaN padding
# 3. **Verification output**: Shows sample counts for validation
# 4. **Better structure**: Direct DataFrame construction
# 5. **Test set consistency**: Same test set across all folds

# **Example output:**
# ```
# Total samples: 500
# Label distribution: {0: 250, 1: 250}

# Train+Val samples: 400, Test samples: 100

# Fold 1: Train=320, Val=80, Test=100, Total=500
# Fold 2: Train=320, Val=80, Test=100, Total=500
# Fold 3: Train=320, Val=80, Test=100, Total=500
# Fold 4: Train=320, Val=80, Test=100, Total=500
# Fold 5: Train=320, Val=80, Test=100, Total=500

# ✓ Generated 5 folds in /dir/to/save/dataset/csvs/your_dataset_name/

# import pandas as pd
# from sklearn.model_selection import train_test_split, StratifiedKFold
# import argparse
# import os

# def Balanced_k_fold_train_val_then_test(args):
#     csv_path = args.csv_path
#     df = pd.read_csv(csv_path)
#     test_ratio = args.test_ratio
#     save_dir = args.save_dir
#     dataset_name = args.dataset_name
#     train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df['label'], random_state=args.seed, shuffle=True)
#     K=args.k
#     skf = StratifiedKFold(n_splits=K, random_state=args.seed, shuffle=True)


#     for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df['label'])):
#         train_fold_df = train_df.iloc[train_index]
#         val_fold_df = train_df.iloc[val_index]
        
#         combined_df = pd.concat([
#             train_fold_df.rename(columns={'slide_path': 'train_slide_path', 'label': 'train_label'}),
#             val_fold_df.rename(columns={'slide_path': 'val_slide_path', 'label': 'val_label'}),
#             test_df.rename(columns={'slide_path': 'test_slide_path', 'label': 'test_label'})
#         ], axis=1)
#         os.makedirs(f'{args.save_dir}/{args.dataset_name}', exist_ok=True)
#         combined_df.to_csv(f'{save_dir}/{dataset_name}/Total_{K}-fold_{dataset_name}_{fold+1}fold.csv', index=False)
    
# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--seed', type=int, default=42)
#     argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
#     argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
#     argparser.add_argument('--test_ratio', type=float, default=0.2) # first select test_ratio of data as test data
#     argparser.add_argument('--k', type=int, default=5) # then split the rest of data into k folds
#     argparser.add_argument('--save_dir', type=str, default='/dir/to/save/dataset/csvs')
#     args = argparser.parse_args()
#     Balanced_k_fold_train_val_then_test(args)