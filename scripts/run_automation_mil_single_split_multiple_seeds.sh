#!/bin/bash

# ======================= GLOBAL PARAMETERS =======================
PRE_PROCESSOR="MUFASA"  # CLAM, TRIDENT, HISTOLAB, MUFASA 
ROOT_PATH="/data_64T_3/Raja/MUFASA/1.WSI_Classification/Datasets"
NUM_EPOCHS=200
LOG_ROOT_DIR="/data_64T_3/Raja/MUFASA/1.WSI_Classification/Classification_Logs_new/${PRE_PROCESSOR}"

# Arrays for iteration
DATASETS=("CAMELYON16")   #  ("TCGA_NSCLC" "CAMELYON16")

# Model list
# For the following models activate wsi_mil_classification environment
MODELS=("MAMBA_MIL")
# MODELS=("MEAN_MIL" "IB_MIL" "AB_MIL" "CLAM_MB_MIL" "DeepAttn_MIL" "DS_MIL" "TRANS_MIL" "DTFD_MIL" "MHIM_MIL" "ILRA_MIL" "DGR_MIL" "MICO_MIL" "AC_MIL" "ADD_MIL" "TDA_MIL")

# Other_MIL - conda activate wsi_mil_classification
# MAMBA_MIL - activate mambamil environment
# MAMBA2D_MIL - activate mambamil_2d environment
# PTC_MIL - activate transformer environment
# for all other models - activate wsi_mil_classification environment

EXTRACTORS=("resnet50_1024_set1_set2_set3_combined")     
# EXTRACTORS=("resnet50_1024_set1" "resnet18") 

DIMS=(1024)   
# DIMS=(1024 512)

# Parallel Arrays for Seeds and Devices (Batch of 5)
SEEDS=(41 1337 2023 3407 9999)
DEVICES=(0 1 2 3 4)

# ======================= EXECUTION LOOP =======================

# 1. Loop through Datasets
for DATASET in "${DATASETS[@]}"; do
    
    # 2. Loop through Feature Extractors (using index to get matching Dimension)
    for i in "${!EXTRACTORS[@]}"; do
        FEATURE_EXTRACTOR="${EXTRACTORS[$i]}"
        FEATURE_DIM="${DIMS[$i]}"
        
        # Construct the specific dataset directory path
        DATA_DIR="${ROOT_PATH}/${DATASET}/${DATASET}_${PRE_PROCESSOR}_${FEATURE_EXTRACTOR}_splits"

        # 3. Loop through Models
        for MODEL_NAME in "${MODELS[@]}"; do
            
            echo "---------------------------------------------------------------------------------"
            echo "STARTING BATCH: ${MODEL_NAME} | ${DATASET} | ${FEATURE_EXTRACTOR} (Dim: ${FEATURE_DIM})"
            echo "Data Dir: ${DATA_DIR}"
            echo "---------------------------------------------------------------------------------"

            # 4. Loop through the batch of Seeds/Devices (Launch 5 jobs in parallel)
            for j in "${!SEEDS[@]}"; do
                SEED="${SEEDS[$j]}"
                DEVICE="${DEVICES[$j]}"

                python train_mil.py \
                --yaml_path configs/${MODEL_NAME}.yaml \
                --options Dataset.DATASET_NAME=${DATASET} \
                          Dataset.feature_extractor="${FEATURE_EXTRACTOR}" \
                          Model.in_dim=${FEATURE_DIM} \
                          Dataset.dataset_root_dir="${DATA_DIR}" \
                          Logs.log_root_dir="${LOG_ROOT_DIR}" \
                          General.seed=${SEED} \
                          General.num_epochs=${NUM_EPOCHS} \
                          General.device=${DEVICE} &
            done

            # Wait for the batch of 5 seeds to finish before starting the next model/config
            wait
            echo "Batch completed for ${MODEL_NAME}."
            echo ""
        done
    done
done

echo "All experiments completed successfully."