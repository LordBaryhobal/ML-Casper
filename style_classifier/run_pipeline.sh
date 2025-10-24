#!/bin/bash
# run_pipeline.sh

set -e

# --- Generate Timestamp ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting pipeline run with Timestamp: $TIMESTAMP"

# --- Define Timestamped Paths ---
MODEL_PATH="models/model_${TIMESTAMP}.pth"
HISTORY_PLOT_PATH="reports/history_${TIMESTAMP}.png"
CM_PLOT_PATH="reports/cm_${TIMESTAMP}.png"

echo "Étape 1: Collecte des données..."
python3 src/01_data_collection.py \
    --output_csv "data/raw/player_games_from_api.csv"

echo "Étape 2: Prétraitement et tokenisation..."
python3 src/02_preprocessing.py \
    --input_csv "data/raw/player_games_from_api.csv" \
    --vocab_file "models/vocab.json" \
    --label_map_file "models/label_map.json" \
    --output_parquet "data/processed/preprocessed_games.parquet"

echo "Étape 3: Entraînement du modèle..."
python3 src/03_model_training.py \
    --processed_data_file "data/processed/preprocessed_games.parquet" \
    --vocab_file "models/vocab.json" \
    --label_map_file "models/label_map.json" \
    # --- Pass Timestamped Paths ---
    --model_output_path "$MODEL_PATH" \
    --history_plot_path "$HISTORY_PLOT_PATH" \
    --cm_plot_path "$CM_PLOT_PATH" \
    
echo "Pipeline run $TIMESTAMP terminé avec succès."
echo "Model saved to: $MODEL_PATH"
echo "Reports saved to: $HISTORY_PLOT_PATH, $CM_PLOT_PATH"