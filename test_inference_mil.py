import os
import sys            
import argparse
import random
import numpy as np

import shutil
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.model_utils import get_model_from_yaml, get_criterion
from utils.yaml_utils import read_yaml, update_config_from_options, change_yaml_by_options  
from utils.loop_utils import val_loop, clam_val_loop, ds_val_loop, dtfd_val_loop
from utils.wsi_utils import WSI_Dataset_Test, CDP_MIL_WSI_Dataset, LONG_MIL_WSI_Dataset

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1: Tee — writes every print() to both the terminal and summary.txt
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    """Mirrors writes to all streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()
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
# ─────────────────────────────────────────────────────────────────────────────
# Main test function
# ─────────────────────────────────────────────────────────────────────────────
def test(args):
    yaml_path = args.yaml_path
    yaml_args = read_yaml(yaml_path)

    # Override the parameters in .yaml config file 
    if args.options:
        yaml_args = update_config_from_options(yaml_args, args.options)

    # ── Apply seed before anything else ──────────────────────────────────────
    seed = yaml_args.General.seed
    set_seed(seed)

    model_name  = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes

    test_log_dir = yaml_args.Logs.log_root_dir
    os.makedirs(test_log_dir, exist_ok=True)

    # ── Open summary.txt and tee stdout into it ────────────────────
    summary_path = os.path.join(test_log_dir, 'summary.txt')
    _log_file    = open(summary_path, 'w', encoding='utf-8')
    sys.stdout   = _Tee(sys.__stdout__, _log_file)

    # ── Save the (options-applied) YAML to the output dir ──────────
    #    Copies the original file first so formatting/comments are preserved,
    #    then patches only the keys that were overridden via --options.
    #    The original file at --yaml_path is NEVER modified.
    saved_yaml_path = os.path.join(test_log_dir, f'config_{model_name}.yaml')
    shutil.copyfile(yaml_path, saved_yaml_path)
    if args.options:
        change_yaml_by_options(saved_yaml_path, args.options)   # patches the copy only

    # ── Copy the dataset CSV into the output dir ───────────────────
    csv_save_path = os.path.join(test_log_dir, os.path.basename(yaml_args.Dataset.dataset_csv_path))
    shutil.copyfile(yaml_args.Dataset.dataset_csv_path, csv_save_path)

    # ── Now it's safe to print (everything goes to terminal + summary.txt) ───
    print('\n----------------INFO----------------\n')
    print(f"MIL-model-yaml path   : {yaml_path}")
    print(f"Dataset name          : {yaml_args.Dataset.DATASET_NAME}") 
    print(f"✓ Global seed set to  : {seed}")
    print(f"Group (test/val)      : {yaml_args.Dataset.group}")
    print(f"Dataset csv path      : {yaml_args.Dataset.dataset_csv_path}")
    print(f"Model weight path     : {args.model_weight_path}")
    print(f"Output directory      : {test_log_dir}")
    print(f"✓ Config saved to     : {saved_yaml_path}")
    print(f"✓ Dataset CSV saved to: {csv_save_path}")
    print(f"✓ Console log saved to: {summary_path}")

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    test_ds = WSI_Dataset_Test(yaml_args.Dataset.dataset_csv_path, yaml_args, yaml_args.Dataset.group)

    inference_mode = not test_ds.is_with_labels()
    if inference_mode:
        print("⚠️  Inference mode: No labels found in CSV. Only predictions will be generated.")

    g = torch.Generator()
    g.manual_seed(seed)

    test_dataloader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=yaml_args.General.get("num_workers", 0),
        worker_init_fn=_worker_init_fn,
        generator=g,
        pin_memory=True,
    )

    device = torch.device(f'cuda:{yaml_args.General.device}')

    # ── Load model ────────────────────────────────────────────────────────────
    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        classifier, attention, dimReduction, attCls = get_model_from_yaml(yaml_args)
        state_dict = torch.load(args.model_weight_path, weights_only=True, map_location=device)
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        model_list = [classifier, attention, dimReduction, attCls]
        model_list = [model.to(device).eval() for model in model_list]
    else:
        mil_model = get_model_from_yaml(yaml_args)
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(
            torch.load(args.model_weight_path, weights_only=True, map_location=device)
        )
        mil_model.eval()

    # ── Helper: single forward pass → logits tensor ───────────────────────────
    def _forward(bag):
        if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
            classifier, attention, dimReduction, attCls = model_list
            slide_pseudo_feat = []
            inputs_pseudo_bags = torch.chunk(bag.squeeze(0), yaml_args.Model.num_Group, dim=0)
            for subFeat_tensor in inputs_pseudo_bags:
                subFeat_tensor  = subFeat_tensor.to(device)
                tmidFeat        = dimReduction(subFeat_tensor)
                tAA             = attention(tmidFeat).squeeze(0)
                tattFeats       = torch.einsum('ns,n->ns', tmidFeat, tAA)
                tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)
                slide_pseudo_feat.append(tattFeat_tensor)
            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
            return attCls(slide_pseudo_feat)['logits']
    
        if yaml_args.General.MODEL_NAME == 'DS_MIL':
            output = mil_model(bag)
            if isinstance(output, dict):
                return output['logits']   # ← dict return, use logits key
            return output[0] if isinstance(output, (tuple, list)) else output
    
        output = mil_model(bag)
    
        if isinstance(output, dict):
            for key in ('logits', 'features', 'output'):
                if key in output:
                    return output[key]
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise ValueError(f"Model returned a dict but no recognised logit key. Keys: {list(output.keys())}")
    
        if isinstance(output, (tuple, list)):
            return output[0]
    
        return output

    # ── 4a. Inference mode ────────────────────────────────────────────────────
    if inference_mode:
        predictions   = []
        probabilities = []
        slide_names   = []

        print("\nRunning inference...")
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                bag    = data[0].to(device).float()
                logits = _forward(bag)
                probs  = torch.softmax(logits, dim=1)
                pred   = torch.argmax(probs, dim=1).cpu().item()
                predictions.append(pred)
                probabilities.append(probs.cpu().numpy()[0].tolist())
                slide_names.append(os.path.basename(test_ds.slide_path_list[i]))

        results_df = pd.DataFrame({
            'slide_path':    test_ds.slide_path_list,
            'slide_name':    slide_names,
            'prediction':    predictions,
            'probabilities': probabilities,
        })

        predictions_path = os.path.join(test_log_dir, f'Predictions_{model_name}.csv')
        results_df.to_csv(predictions_path, index=False)

        print(f"\n✓ Predictions saved to        : {predictions_path}")
        print(f"Total slides                  : {len(predictions)}")
        print(f"Prediction distribution       : {pd.Series(predictions).value_counts().to_dict()}")

    # ── 4b. Test mode with labels ─────────────────────────────────────────────
    else:
        criterion = get_criterion(yaml_args.Model.criterion)

        if yaml_args.General.MODEL_NAME in ['CLAM_MB_MIL', 'CLAM_SB_MIL']:
            bag_weight = yaml_args.Model.bag_weight
            test_loss, test_metrics = clam_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight)
        elif yaml_args.General.MODEL_NAME == 'DS_MIL':
            test_loss, test_metrics = ds_val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
            test_loss, test_metrics = dtfd_val_loop(device, num_classes, model_list, test_dataloader, criterion,
                                                    yaml_args.Model.num_Group, yaml_args.Model.grad_clipping,
                                                    yaml_args.Model.distill, yaml_args.Model.total_instance,)
        else:
            test_loss, test_metrics = val_loop(device, num_classes, mil_model, test_dataloader, criterion)

        print("\nCollecting individual slide probabilities...")
        slide_names = []
        all_probs   = []
        all_preds   = []
        true_labels = []

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                bag   = data[0].to(device).float()
                label = data[1].item()

                logits = _forward(bag)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred   = int(np.argmax(probs))

                slide_names.append(os.path.basename(test_ds.slide_path_list[i]))
                all_probs.append(probs.tolist())
                all_preds.append(pred)
                true_labels.append(label)

        detailed_results_df = pd.DataFrame({'slide_name':slide_names,'true_label':true_labels,'prediction':all_preds,'probabilities':all_probs,})

        detailed_csv_path = os.path.join(test_log_dir, f'Detailed_Probabilities_{model_name}.csv')
        detailed_results_df.to_csv(detailed_csv_path, index=False)

        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('\n------TEST/VALIDATION METRICS------\n') 
        print(f'{FAIL}Test_Loss   :{ENDC} {test_loss}')
        print(f'{FAIL}Test_Metrics:{ENDC} {test_metrics}')
        print(f"\n✓ Detailed probabilities saved to: {detailed_csv_path}")
        print('\n----------------END----------------\n')

    # ── CHANGE 1: restore stdout and flush/close the log file ────────────────
    sys.stdout = sys.__stdout__
    _log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',         type=str, required=True, help='path to MIL-model-yaml file') 
    parser.add_argument('--model_weight_path', type=str, required=True, help='path to model weights') 
    parser.add_argument('--options', nargs='+', help='override yaml config: key=value pairs, e.g. General.seed=0')
    args = parser.parse_args()
    test(args)