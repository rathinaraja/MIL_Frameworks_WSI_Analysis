#!/bin/bash

# ======================= CONDA INITIALIZATION =======================
# (Kept identical to your source)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Error: Could not find conda.sh. Please ensure conda is in your PATH."
        exit 1
    fi
fi

# ======================= GLOBAL PARAMETERS =======================
PRE_PROCESSOR="TRIDENT"   
# CLAM, TRIDENT, HISTOLAB, MUFASA 
ROOT_PATH="/data_64T_3/Raja/MUFASA/1.WSI_Classification/Datasets"
NUM_EPOCHS=200
LOG_ROOT_DIR="/data_64T_3/Raja/MUFASA/1.WSI_Classification/AUROC_Analysis/Output/${PRE_PROCESSOR}" 
SEED=42

# DATASETS and EXTRACTORS must have the same number of elements
DATASETS=("TCGA_COAD_READ")   #("TCGA_NSCLC" "STANFORD_793" "TCGA_COAD_READ")
EXTRACTORS=("resnet50_1024")   
# ("resnet_1024" "uni")  # for mufasa, example, resnet_1024_set1, resnet_1024_set1_set2_set3_combined 
DIMS=(1024)                  # (1024 1024 1024) 

DEVICES=(0 1 2 3 4 5 6 7) # Your specific GPU list

# ======================= Eeach GPU gets a MIL model =======================
train_model_batch() {
    local CURRENT_ENV=$(conda info | grep "active environment" | cut -d : -f 2 | tr -d ' ')
    echo "Current Environment: $CURRENT_ENV"

    local gpu_idx=0
    local num_gpus=${#DEVICES[@]}

    # 1st Loop: Datasets
    for DATASET in "${DATASETS[@]}"; do
        
        # 2nd Loop: Feature Extractors (using index to pick Dim)
        for i in "${!EXTRACTORS[@]}"; do
            FEATURE_EXTRACTOR="${EXTRACTORS[$i]}"
            FEATURE_DIM="${DIMS[$i]}"  # Automatically picks dimension based on extractor index

            # 3rd Loop: MIL Models
            for MODEL_NAME in "${MODELS[@]}"; do
                
                # Pick the next GPU device
                DEVICE="${DEVICES[$gpu_idx]}"
                
                # Construct the dataset root directory
                DATASET_ROOT="${ROOT_PATH}/${DATASET}/${DATASET}_${PRE_PROCESSOR}_${FEATURE_EXTRACTOR}_splits"

                echo "-------------------------------------------------------"
                echo "GPU ${DEVICE} | ${MODEL_NAME}"
                echo "Data: ${DATASET} | Extractor: ${FEATURE_EXTRACTOR} | Dim: ${FEATURE_DIM}"
                echo "-------------------------------------------------------"

                # Execute in background
                python /home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/train_mil.py \
                    --yaml_path "/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/configs/${MODEL_NAME}.yaml" \
                    --options "Dataset.DATASET_NAME=${DATASET}" \
                              "Dataset.feature_extractor=${FEATURE_EXTRACTOR}" \
                              "Model.in_dim=${FEATURE_DIM}" \
                              "Dataset.dataset_root_dir=${DATASET_ROOT}" \
                              "Logs.log_root_dir=${LOG_ROOT_DIR}" \
                              "General.seed=${SEED}" \
                              "General.num_epochs=${NUM_EPOCHS}" \
                              "General.device=${DEVICE}" &

                # GPU Management
                ((gpu_idx++))

                # If all GPUs are busy, wait for them to finish before starting next set
                if [ $gpu_idx -eq $num_gpus ]; then
                    echo ">>> Batch full. Waiting for GPUs [${DEVICES[*]}] to finish..."
                    wait
                    gpu_idx=0
                fi
            done
        done
    done
    wait # Final wait for the last batch
}
# ======================= EXECUTION FLOW =======================

# echo "Activating: mambamil"
# conda activate mambamil

# Define the models you want to run for this specific environment
# MODELS=("MEAN_MIL" "IB_MIL" "AB_MIL" "CLAM_MB_MIL" "DeepAttn_MIL" "DS_MIL" "TRANS_MIL" "DTFD_MIL" "MHIM_MIL" "ILRA_MIL" "DGR_MIL" "MICO_MIL" "AC_MIL" "ADD_MIL" "TDA_MIL")
MODELS=("DS_MIL")

train_model_batch

echo "======================================================="
echo "All assigned models have completed their internal folds."
echo "======================================================="