#!/bin/bash

# echo "Starting MEAN_MIL..."

# --- MEAN_MIL --- 
python train_mil.py \
    --yaml_path configs/MEAN_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=0 &

# --- IB_MIL --- 
python train_mil.py \
    --yaml_path configs/IB_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=1 &


# --- AB_MIL --- 
python train_mil.py \
    --yaml_path configs/AB_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=2 &

# --- CLAM_MB_MIL --- 
python train_mil.py \
    --yaml_path configs/CLAM_MB_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=3 &

# --- DeepAttn_MIL --- 
python train_mil.py \
    --yaml_path configs/DeepAttn_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=4 &

# --- DS_MIL --- 
python train_mil.py \
    --yaml_path configs/DS_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=5 &


# --- TRANS_MIL --- 
python train_mil.py \
    --yaml_path configs/TRANS_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=6 &

# --- DTFD_MIL --- 
python train_mil.py \
    --yaml_path configs/DTFD_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=7 &

wait

# --- MHIM_MIL --- 
python train_mil.py \
    --yaml_path configs/MHIM_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=0 &

# --- ILRA_MIL --- 
python train_mil.py \
    --yaml_path configs/ILRA_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=1 &


# --- DGR_MIL --- 
python train_mil.py \
    --yaml_path configs/DGR_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=2 &

# --- MICO_MIL --- 
python train_mil.py \
    --yaml_path configs/MICO_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=3 &

# --- AC_MIL --- 
python train_mil.py \
    --yaml_path configs/AC_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=4 &

# --- ADD_MIL --- 
python train_mil.py \
    --yaml_path configs/ADD_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=5 &


# --- TDA_MIL --- 
python train_mil.py \
    --yaml_path configs/TDA_MIL.yaml \
    --options Dataset.DATASET_NAME="TCGA_NSCLC" \
              Dataset.feature_extractor="resnet50_1024_set1_set2_set3_combined" \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir="/data_55T_2/Raja/Datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_set2_set3_combined_splits" \
              Logs.log_root_dir="/data_55T_2/Raja/CLASSIFICATION/MUFASA" \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=6 

wait

echo "All jobs completed."