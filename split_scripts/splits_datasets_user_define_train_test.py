from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

def Balanced_Train_Test(args):
    df = pd.read_csv(args.csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Split ratios: Train={args.train_ratio}, Test={args.test_ratio}\n")
    
    X = df['slide_path']
    y = df['label']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=args.train_ratio, 
        stratify=y, 
        random_state=args.seed,
        shuffle=True
    )
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Pad to same length
    max_len = max(len(X_train), len(X_test))
    
    result = pd.DataFrame({
        'train_slide_path': X_train.reindex(range(max_len)),
        'train_label': y_train.reindex(range(max_len)),
        'val_slide_path': pd.Series([pd.NA] * max_len),
        'val_label': pd.Series([pd.NA] * max_len),
        'test_slide_path': X_test.reindex(range(max_len)),
        'test_label': y_test.reindex(range(max_len))
    })
    
    result.to_csv(args.save_path, index=False)
    
    # Verification
    n_train = result['train_slide_path'].notna().sum()
    n_test = result['test_slide_path'].notna().sum()
    n_total = n_train + n_test
    
    print(f"Train={n_train} ({n_train/n_total*100:.1f}%), "
          f"Test={n_test} ({n_test/n_total*100:.1f}%), "
          f"Total={n_total}")
    print(f"\n✓ Saved to {args.save_path}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--save_path', type=str, default='/path/to/your/save-path.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--train_ratio', type=float, default=0.7)
    argparser.add_argument('--test_ratio', type=float, default=0.3)
    args = argparser.parse_args()
    
    assert abs(args.train_ratio + args.test_ratio - 1.0) < 1e-6, \
        'train_ratio + test_ratio must equal 1'
    
    Balanced_Train_Test(args)
    
# ```

# **Key improvements:**

# 1. **Proper indexing**: Reset indices after split
# 2. **Clean padding**: Use `.reindex()` for consistent NaN padding
# 3. **Empty val columns**: Use `pd.NA` for proper empty columns (instead of empty Series)
# 4. **Better assertion**: Use floating point comparison
# 5. **Verification output**: Shows actual sample counts and percentages
# 6. **Simplified X extraction**: Direct `df['slide_path']`

# **Example output:**
# ```
# Total samples: 1000
# Label distribution: {0: 500, 1: 500}
# Split ratios: Train=0.7, Test=0.3

# Train=700 (70.0%), Test=300 (30.0%), Total=1000

# ✓ Saved to /path/to/your/save-path.csv
# ```

# **Output CSV structure:**
# ```
# train_slide_path,train_label,val_slide_path,val_label,test_slide_path,test_label
# /path/slide1.pt,1,,,/path/slide701.pt,0
# /path/slide2.pt,0,,,/path/slide702.pt,1
# ...

# from sklearn.model_selection import StratifiedKFold
# import pandas as pd
# import numpy as np
# import os
# import argparse
    
# from sklearn.model_selection import train_test_split

# def Balanced_Train_Test(args):
#     df = pd.read_csv(args.csv_path)

#     X = df.drop('label', axis=1)  
#     y = df['label']

#     train_size = args.train_ratio 
#     test_size = args.test_ratio  

#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=args.seed, shuffle=True)

#     result = pd.DataFrame({
#         'train_slide_path': pd.Series(X_train.values.flatten()),
#         'train_label': pd.Series(y_train.values),
#         'val_slide_path': pd.Series([]),
#         'val_label': pd.Series([]),
#         'test_slide_path': pd.Series(X_temp.values.flatten()),
#         'test_label': pd.Series(y_temp.values)
#     })

#     result.to_csv(args.save_path, index=False)


# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--seed', type=int, default=42)
#     argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
#     argparser.add_argument('--save_path', type=str, default='/path/to/your/save-path.csv')
#     argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
#     argparser.add_argument('--train_ratio', type=float, default=0.7)
#     argparser.add_argument('--test_ratio', type=float, default=0.3)
#     args = argparser.parse_args()
#     assert args.train_ratio + args.test_ratio == 1 , print('train_ratio + test_ratio must be equal to 1')
#     Balanced_Train_Test(args)

