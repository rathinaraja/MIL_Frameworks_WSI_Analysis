import os
import argparse 
import random
import numpy as np
import torch

from utils.yaml_utils import read_yaml, update_config_from_options
from utils.general_utils import get_time, merge_k_fold_logs
from process.process_all import process

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Determinism helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(arg):
    yaml_path = arg.yaml_path
    print(f"MIL-yaml path: {yaml_path}")
    args = read_yaml(yaml_path)
    # dinamically update the config file with the options
    if arg.options:
        args = update_config_from_options(args,arg.options) 

    # ── Apply seed before anything else ──────────────────────────────────────
    seed = args.General.seed
    set_seed(seed)
    
    if args.Dataset.dataset_csv_path != "None":
        '''
        None-fold split
        ''' 
        log_root_dir = args.Logs.log_root_dir
        os.makedirs(log_root_dir,exist_ok=True)
        sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME,args.Dataset.feature_extractor)
        os.makedirs(sub_dir,exist_ok=True)
        args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{get_time()}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}_{args.Dataset.feature_extractor}')
        process(args,yaml_path,arg.options)

    else:
        '''
        k-fold split
        ''' 
        dataset_root_dir = args.Dataset.dataset_root_dir
        k_fold_csv_paths = sorted([os.path.join(dataset_root_dir,path) for path in os.listdir(dataset_root_dir)])
        k_fold_csv_paths = [p for p in k_fold_csv_paths if p.endswith('.csv')]

        process_time = get_time()
        for k_idx,k_fold_csv_path in enumerate(k_fold_csv_paths):
            args.Dataset.dataset_csv_path = k_fold_csv_path
            now_fold = k_idx+1
            args.Dataset.now_fold = now_fold
            log_root_dir = args.Logs.log_root_dir
            os.makedirs(log_root_dir,exist_ok=True)
            sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME,args.Dataset.feature_extractor)
            os.makedirs(sub_dir,exist_ok=True)
            if now_fold != None:
                fold_dir = f'fold_{now_fold}'
                args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}_{args.Dataset.feature_extractor}/{fold_dir}')
            os.makedirs(args.Logs.now_log_dir,exist_ok=True)
            process(args,yaml_path,arg.options)
            print(f'K-Fold:{k_idx+1} Done!')
        fold_total_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}_{args.Dataset.feature_extractor}')    
        merge_k_fold_logs(fold_total_log_dir,args.General.process_pipeline)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/path/to/your/yaml',help='path to MIL-yaml file')
    parser.add_argument('--options',nargs='+',help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the yaml config file')
    arg = parser.parse_args()
    main(arg)
 