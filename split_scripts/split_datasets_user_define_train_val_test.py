from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

def Balanced_Train_Val_Test(args):
    df = pd.read_csv(args.csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}\n")
    
    X = df['slide_path']
    y = df['label']
    
    # First split: separate train from (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=args.train_ratio, 
        stratify=y, 
        random_state=args.seed,
        shuffle=True
    )
    
    # Second split: separate val from test
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        train_size=val_ratio_adjusted, 
        stratify=y_temp, 
        random_state=args.seed,
        shuffle=True
    )
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Pad to same length
    max_len = max(len(X_train), len(X_val), len(X_test))
    
    result = pd.DataFrame({
        'train_slide_path': X_train.reindex(range(max_len)),
        'train_label': y_train.reindex(range(max_len)),
        'val_slide_path': X_val.reindex(range(max_len)),
        'val_label': y_val.reindex(range(max_len)),
        'test_slide_path': X_test.reindex(range(max_len)),
        'test_label': y_test.reindex(range(max_len))
    })
    
    result.to_csv(args.save_path, index=False)
    
    # Verification
    n_train = result['train_slide_path'].notna().sum()
    n_val = result['val_slide_path'].notna().sum()
    n_test = result['test_slide_path'].notna().sum()
    n_total = n_train + n_val + n_test
    
    print(f"Train={n_train} ({n_train/n_total*100:.1f}%), "
          f"Val={n_val} ({n_val/n_total*100:.1f}%), "
          f"Test={n_test} ({n_test/n_total*100:.1f}%), "
          f"Total={n_total}")
    print(f"\n✓ Saved to {args.save_path}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--save_path', type=str, default='/path/to/your/save-path.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--train_ratio', type=float, default=0.6)
    argparser.add_argument('--val_ratio', type=float, default=0.2)
    argparser.add_argument('--test_ratio', type=float, default=0.2)
    args = argparser.parse_args()
    
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        'train_ratio + val_ratio + test_ratio must equal 1'
    
    Balanced_Train_Val_Test(args)
    
# ```

# **Key improvements:**

# 1. **Fixed split logic**: Use `train_size` instead of `test_size` in second split for correct proportions
# 2. **Proper indexing**: Reset indices after each split
# 3. **Clean padding**: Use `.reindex()` for consistent NaN padding
# 4. **Better assertion**: Use floating point comparison for ratio sum
# 5. **Verification output**: Shows actual sample counts and percentages
# 6. **Simplified X extraction**: Direct `df['slide_path']` instead of dropping label column

# **Example output:**
# ```
# Total samples: 1000
# Label distribution: {0: 500, 1: 500}
# Split ratios: Train=0.6, Val=0.2, Test=0.2

# Train=600 (60.0%), Val=200 (20.0%), Test=200 (20.0%), Total=1000

# ✓ Saved to /path/to/your/save-path.csv

# from sklearn.model_selection import StratifiedKFold
# import pandas as pd
# import numpy as np
# import os
# import argparse
    
# from sklearn.model_selection import train_test_split

# def Balanced_Train_Val_Test(args):
#     df = pd.read_csv(args.csv_path)

#     X = df.drop('label', axis=1)  
#     y = df['label']

#     train_size = args.train_ratio
#     test_size = args.test_ratio  
#     val_size = args.val_ratio  

#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=args.seed)

#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + val_size), stratify=y_temp, random_state=args.seed, shuffle=True)

#     result = pd.DataFrame({
#         'train_slide_path': pd.Series(X_train.values.flatten()),
#         'train_label': pd.Series(y_train.values),
#         'val_slide_path': pd.Series(X_val.values.flatten()),
#         'val_label': pd.Series(y_val.values),
#         'test_slide_path': pd.Series(X_test.values.flatten()),
#         'test_label': pd.Series(y_test.values)
#     })

#     result.to_csv(args.save_path, index=False)


# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--seed', type=int, default=42)
#     argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
#     argparser.add_argument('--save_path', type=str, default='/path/to/your/save-path.csv')
#     argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
#     argparser.add_argument('--train_ratio', type=float, default=0.6)
#     argparser.add_argument('--val_ratio', type=float, default=0.2)
#     argparser.add_argument('--test_ratio', type=float, default=0.2)
#     args = argparser.parse_args()
#     assert args.train_ratio + args.val_ratio + args.test_ratio == 1 , print('train_ratio + val_ratio + test_ratio must be equal to 1')
#     Balanced_Train_Val_Test(args)

