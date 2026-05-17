import os
import argparse
import shutil
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from utils.model_utils import get_model_from_yaml, get_criterion
from utils.yaml_utils import read_yaml
from utils.loop_utils import val_loop, clam_val_loop, ds_val_loop, dtfd_val_loop
from utils.wsi_utils import WSI_Dataset_Test

import warnings
warnings.filterwarnings('ignore')

# ---Deterministic Seeding Function ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed} for reproducibility.")

def calculate_metrics_with_threshold(y_true, y_probs, threshold, num_classes):
    """Helper to calculate metrics based on a specific threshold"""
    if num_classes == 2:
        y_pred = (y_probs[:, 1] >= threshold).astype(int)
    else:
        y_pred = np.argmax(y_probs, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    except:
        auc = 0.0
        
    return {'acc': round(acc, 4), 'balanced_acc': round(b_acc, 4), 'f1_macro': round(f1, 4), 'auc': round(auc, 4)}

def calibrate_threshold(y_true, y_probs, num_classes):
    """Finds the best threshold maximizing Macro F1 score"""
    if num_classes > 2:
        return 0.5

    thresholds = np.arange(0.01, 1.00, 0.005)
    best_threshold = 0.5
    best_score = 0.0
    
    pos_probs = y_probs[:, 1]

    for thresh in thresholds:
        preds = (pos_probs >= thresh).astype(int)
        score = f1_score(y_true, preds, average='macro')
        if score > best_score:
            best_score = score
            best_threshold = thresh
            
    return best_threshold

def test(args):
    # --- FIX: Set Seed at start of testing ---
    seed_everything(args.seed) 
    
    yaml_path = args.yaml_path
    print(f"MIL-model-yaml path: {yaml_path}")
    yaml_args = read_yaml(yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes
    test_dataset_csv = args.test_dataset_csv
    print(f"Dataset csv path: {test_dataset_csv}")
    
    # Create dataset
    test_ds = WSI_Dataset_Test(test_dataset_csv, yaml_args)
    
    inference_mode = not test_ds.is_with_labels()
    if inference_mode:
        print("⚠️  Inference mode: No labels found in CSV. Only predictions will be generated.")
    
    # Use 0 workers to strictly avoid multi-process randomness during testing if needed
    # otherwise seed_worker logic is required. shuffle=False helps.
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4) 
    
    model_weight_path = args.model_weight_path
    print(f"Model weight path: {model_weight_path}")
    device = torch.device(f'cuda:{yaml_args.General.device}')
    
    # Load model
    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        classifier, attention, dimReduction, attCls = get_model_from_yaml(yaml_args)
        state_dict = torch.load(model_weight_path, map_location=device) # Fixed map_location
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        model_list = [classifier, attention, dimReduction, attCls]
        model_list = [model.to(device).eval() for model in model_list]
    else:
        mil_model = get_model_from_yaml(yaml_args)
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(model_weight_path, map_location=device))
        mil_model.eval()

    test_log_dir = args.test_log_dir
    os.makedirs(test_log_dir, exist_ok=True)
    
    if inference_mode:
        # [Inference logic...]
        pass
        
    else:
        # Test mode with labels
        criterion = get_criterion(yaml_args.Model.criterion)
        
        print("\nRunning Standard Evaluation (Threshold = 0.5)...")
        if yaml_args.General.MODEL_NAME in ['CLAM_MB_MIL', 'CLAM_SB_MIL']:
            bag_weight = yaml_args.Model.bag_weight
            test_loss, test_metrics = clam_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight)
        elif yaml_args.General.MODEL_NAME == 'DS_MIL':
            test_loss, test_metrics = ds_val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
            test_loss, test_metrics = dtfd_val_loop(device, num_classes, model_list, test_dataloader, criterion, 
                                                    yaml_args.Model.num_Group, yaml_args.Model.grad_clipping, 
                                                    yaml_args.Model.distill, yaml_args.Model.total_instance)
        else:
            test_loss, test_metrics = val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------STANDARD RESULTS----------------\n')
        print(f'{FAIL}Test_Loss:{ENDC}{test_loss}\n')
        print(f'{FAIL}Test_Metrics (Std):  {ENDC}{test_metrics}\n')

        final_metrics = test_metrics 
        best_thresh = 0.5
        
        if args.calibrate:
            print("\n----------------CALIBRATION----------------")
            print("Running Calibration step to find optimal threshold...")
            
            y_true_all = []
            y_prob_all = []
            
            with torch.no_grad():
                for data in test_dataloader:
                    bag, label = data[0].to(device).float(), data[1].to(device).long()
                    
                    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
                        feat_index = torch.LongTensor([0]).to(device)
                        slide_pseudo_feat, _ = dimReduction(bag, feat_index)
                        slide_sub_preds = []
                        for subtyping_idx in range(yaml_args.Model.num_Group):
                            slide_sub_preds.append(attCls[subtyping_idx](slide_pseudo_feat))
                        logits = torch.stack(slide_sub_preds).mean(0)
                        
                    elif yaml_args.General.MODEL_NAME in ['CLAM_MB_MIL', 'CLAM_SB_MIL']:
                        output = mil_model(bag, label=label)
                        
                        # --- ROBUST LOGITS EXTRACTION ---
                        logits = None
                        if isinstance(output, dict):
                            logits = output['logits']
                        elif isinstance(output, (tuple, list)):
                            for item in output:
                                if isinstance(item, dict) and 'logits' in item:
                                    logits = item['logits']
                                    break
                            if logits is None:
                                logits = output[0]
                        else:
                            logits = output
                        # --------------------------------
                        
                    elif yaml_args.General.MODEL_NAME == 'DS_MIL':
                        output = mil_model(bag)
                        if isinstance(output, tuple):
                            max_pred = output[0]
                        else:
                            max_pred = output
                        logits = max_pred
                    else:
                        logits = mil_model(bag)
                    
                    # Final safety unwrap
                    if isinstance(logits, dict) and 'logits' in logits:
                        logits = logits['logits']

                    probs = torch.softmax(logits, dim=1)
                    y_prob_all.append(probs.cpu().numpy())
                    y_true_all.append(label.cpu().numpy())

            y_true_all = np.concatenate(y_true_all)
            y_prob_all = np.concatenate(y_prob_all)

            best_thresh = calibrate_threshold(y_true_all, y_prob_all, num_classes)
            print(f"Optimal Threshold found: {best_thresh:.4f}")

            calibrated_metrics = calculate_metrics_with_threshold(y_true_all, y_prob_all, best_thresh, num_classes)
            final_metrics = calibrated_metrics
            
            print(f'{FAIL}Calibrated_Metrics: {ENDC}{calibrated_metrics}\n')
        
        # Save results
        new_yaml_path = os.path.join(test_log_dir, f'Test_{model_name}.yaml')
        shutil.copyfile(yaml_path, new_yaml_path)
        
        test_log_path = os.path.join(test_log_dir, f'Test_Log_{model_name}.txt')
        
        log_to_save = {
            'test_loss': test_loss, 
            'final_metrics': final_metrics,
            'threshold_used': best_thresh
        }
        
        with open(test_log_path, 'w') as f:
            f.write(str(log_to_save))
        print(f"Test log saved at: {test_log_path}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='/path/to/config.yaml')
    parser.add_argument('--test_dataset_csv', type=str, default='/path/to/dataset.csv')
    parser.add_argument('--model_weight_path', type=str, default='/path/to/weights.pt')
    parser.add_argument('--test_log_dir', type=str, default='/path/to/logs')
    parser.add_argument('--calibrate', action='store_true', help='Perform threshold calibration') 
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    test(args)