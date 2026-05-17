#!/bin/bash

# ======================= CONFIGURATION =======================
# 1. Common Paths
PROJECT_ROOT="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification"
DATASET_ROOT="${PROJECT_ROOT}/datasets/CAMELYON16_Test"
LOGS_ROOT="${PROJECT_ROOT}/logs"
TEST_LOG_ROOT="${LOGS_ROOT}/TEST"

# 2. Model Config
YAML_PATH="configs/MEAN_MIL.yaml"

# 3. Lists to Iterate Over
SEEDS=(41 1337 2023 3407 9999)
PREPROCESSORS=("CLAM" "TRIDENT" "HISTOLAB" "MUFASA")

# 4. Map Preprocessors to their specific Feature Extractor folder names
declare -A FEATURE_EXTRACTORS
FEATURE_EXTRACTORS=(
    ["CLAM"]="resnet50_1024"
    ["TRIDENT"]="resnet50_1024"
    ["HISTOLAB"]="resnet50_1024"
    ["MUFASA"]="resnet50_1024_set1"
)

# ======================= EXECUTION LOOP =======================

echo "Starting Batch Calibration..."

for SEED in "${SEEDS[@]}"; do
    echo "========================================================"
    echo " Processing Seed: ${SEED}"
    echo "========================================================"

    for METHOD in "${PREPROCESSORS[@]}"; do
        
        # 1. Get feature extractor name for this method
        EXTRACTOR="${FEATURE_EXTRACTORS[$METHOD]}"
        
        # 2. Construct Dataset CSV Path
        CSV_FILE="${DATASET_ROOT}/CAMELYON16_${METHOD}_${EXTRACTOR}_splits_test.csv"

        # 3. Auto-locate the Model Weight Path
        #    We search inside the logs folder for a path matching the seed and 'Best_EPOCH'
        SEARCH_DIR="${LOGS_ROOT}/${METHOD}/CAMELYON16/DS_MIL/${EXTRACTOR}"
        
        # Finds the first .pth file that matches the seed pattern
        MODEL_PATH=$(find "${SEARCH_DIR}" -type f -path "*seed_${SEED}_*/fold_1/Best_EPOCH_*.pth" | head -n 1)

        # 4. Safety Check: Skip if model not found
        if [ -z "$MODEL_PATH" ]; then
            echo "⚠️  WARNING: Model not found for ${METHOD} (Seed ${SEED}). Skipping..."
            continue
        fi

        # 5. Define Output Directory
        OUTPUT_DIR="${TEST_LOG_ROOT}/${METHOD}"
        mkdir -p "$OUTPUT_DIR"

        echo "--------------------------------------------------------"
        echo " Method: ${METHOD}"
        echo " Model:  $(basename "$MODEL_PATH")"
        echo "--------------------------------------------------------"

        # 6. Run the Python Command
        python test_mil_calibration.py \
            --yaml_path "$YAML_PATH" \
            --test_dataset_csv "$CSV_FILE" \
            --model_weight_path "$MODEL_PATH" \
            --test_log_dir "$OUTPUT_DIR" \
            --seed "$SEED" \
            --calibrate

    done
done

echo "========================================================"
echo "✅ All Calibration Runs Completed."