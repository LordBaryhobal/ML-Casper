# src/config.py

"""
Fichier de configuration centralisé pour les paramètres du projet.
"""
# --- Paramètres de Collecte de Données ---
PLAYER_CONFIG = {
    'athena-pallada': 'Baadur Jobava',
    'Pap-G': 'Pap Gyula',
    'OhanyanEminChess': 'Emin Ohanyan',
    'Yarebore': 'Paweł Teclaf',
}
GAMES_PER_ROLE = 3000
# Calculez la constante dérivée directement ici
TOTAL_GAMES_TO_FETCH = GAMES_PER_ROLE * 3 # pour être sur d'avoir assez de parties

# --- Paramètres de Données et Séquence (Partagés) ---
# Doit être le même pour le preprocessing et l'entraînement
SEQUENCE_LENGTH = 60

# --- Hyperparamètres du Modèle (Architecture) ---
D_MODEL = 768
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 8
DIM_FEEDFORWARD = 3072
DROPOUT = 0.1

# --- Hyperparamètres d'Entraînement (Défauts) ---
# Ces valeurs seront utilisées comme défauts pour argparse
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
TEST_SPLIT_SIZE = 0.2
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 5
