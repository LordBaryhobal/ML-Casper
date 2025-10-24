# --- LE PSEUDO À TESTER ---
from pathlib import Path


TARGET_USERNAME = 'Erdosz'
NUM_GAMES_TO_TEST = 100 # Combien de parties analyser pour ce joueur

# --- Fichiers Artefacts (de 02 & 03) ---
MODEL_DIR = Path(__file__).parent.parent.parent / "style_model"
MODEL_FILE = MODEL_DIR / 'best_style_transformer_model.pth'
VOCAB_FILE = MODEL_DIR / 'vocab.json'
LABEL_MAP_FILE = MODEL_DIR / 'label_map.json'

# --- Hyperparamètres du Modèle (DOIVENT correspondre à l'entraînement) ---
SEQUENCE_LENGTH = 60
D_MODEL = 512
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
BATCH_SIZE = 64 # Peut être plus grand pour l'inférence

# --- Logique de 02-preprocessing ---
NUM_MOVES_PER_CHUNK = SEQUENCE_LENGTH - 2 
