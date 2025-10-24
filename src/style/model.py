import json

import numpy as np
import torch

from src.style.config import (D_MODEL, DIM_FEEDFORWARD, DROPOUT,
                              LABEL_MAP_FILE, MODEL_FILE, NUM_ENCODER_LAYERS,
                              NUM_HEADS, NUM_MOVES_PER_CHUNK, SEQUENCE_LENGTH,
                              VOCAB_FILE)
from src.style.encoding import StyleTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class StyleAnalyser:
    def __init__(self) -> None:
        self.move_to_id: dict
        self.label_to_player: dict
        self.model: StyleTransformer
        self.load()

    def load(self):
        try:
            with open(VOCAB_FILE, 'r') as f:
                self.move_to_id = json.load(f)
            VOCAB_SIZE = len(self.move_to_id)
            print(f"Vocabulaire chargé (taille={VOCAB_SIZE})")

            with open(LABEL_MAP_FILE, 'r') as f:
                player_to_label = json.load(f)
            NUM_CLASSES = len(player_to_label)
            self.label_to_player = {v: k for k, v in player_to_label.items()}
            print(f"Map des labels chargée (classes={NUM_CLASSES})")

            # Instancier l'architecture du modèle
            self.model = StyleTransformer(
                vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, d_model=D_MODEL,
                nhead=NUM_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS,
                dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
            )

            # Charger les poids entraînés
            self.model.load_state_dict(
                torch.load(MODEL_FILE, map_location=device))
            self.model.to(device)
            self.model.eval()  # !! TRÈS IMPORTANT: Passer en mode évaluation
            print(f"Modèle chargé depuis {MODEL_FILE} et mis en mode .eval()")

        except FileNotFoundError as e:
            print(f"Erreur: Fichier artefact manquant : {e}")
            print(
                "Veuillez vous assurer que vocab.json, label_map.json et le .pth sont présents.")

    def infer(self, game_data) -> dict[str, float]:
        tokenized_chunks = self.tokenize_game(game_data)
        tensor = torch.tensor(tokenized_chunks, dtype=torch.long)

        with torch.no_grad():
            inputs = tensor.to(device)
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        
        avg_probabilities = np.mean(probabilities, axis=0)
        sorted_indices = np.argsort(avg_probabilities)[::-1]  # Décroissant
        result: dict[str, float] = {}

        for i in sorted_indices:
            player_name = self.label_to_player[i]
            probability = avg_probabilities[i] * 100
            print(f"  - {player_name}: {probability:.2f}%")
            result[player_name] = float(probability)

        most_likely_player = self.label_to_player[sorted_indices[0]]
        confidence = avg_probabilities[sorted_indices[0]] * 100
        print(f"Most similar: '{most_likely_player}' ({confidence:.2f}%)")
        return result

    def tokenize_game(self, game_data):
        processed_chunks = []

        moves_str, color = game_data
        moves_list = moves_str.split(' ')

        color_token = '[WHITE]' if color == 'WHITE' else '[BLACK]'

        for i in range(0, len(moves_list), NUM_MOVES_PER_CHUNK):
            move_chunk = moves_list[i: i + NUM_MOVES_PER_CHUNK]

            token_list = ['[CLS]', color_token] + move_chunk

            # Padding
            if len(token_list) < SEQUENCE_LENGTH:
                padding_needed = SEQUENCE_LENGTH - len(token_list)
                token_list.extend(['[PAD]'] * padding_needed)

            token_list = token_list[:SEQUENCE_LENGTH]
            token_ids = [self.move_to_id.get(token, 1) for token in token_list]

            if len(token_ids) == SEQUENCE_LENGTH:
                processed_chunks.append(token_ids)
        return processed_chunks
