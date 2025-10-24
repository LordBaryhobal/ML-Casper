# src/02_preprocessing.py
import polars as pl
import json
from tqdm import tqdm
import argparse
import logging
import os
import random

# --- Constantes ---
from config import SEQUENCE_LENGTH


def main(input_csv, vocab_file, label_map_file, output_parquet):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # S'assurer que les répertoires de sortie existent
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    os.makedirs(os.path.dirname(label_map_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)

    try:
        df_raw = pl.read_csv(input_csv)
        logging.info(f"Loaded {len(df_raw)} raw games from {input_csv}.")
    except FileNotFoundError:
        logging.error(f"Error: {input_csv} not found.")
        return
    
    # --- 1. Construction du vocabulaire ---
    all_moves_set = set()
    for moves_str in df_raw['moves']:
        all_moves_set.update(moves_str.split(' '))
    logging.info(f"Found {len(all_moves_set)} unique moves.")

    move_to_id = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[WHITE]': 3, '[BLACK]': 4}
    id_to_move = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[WHITE]', 4: '[BLACK]'}

    for i, move in enumerate(all_moves_set):
        move_id = i + 5 
        move_to_id[move] = move_id
        id_to_move[move_id] = move_id

    vocab_size = len(move_to_id)
    logging.info(f"Total vocabulary size (including special tokens): {vocab_size}")

    with open(vocab_file, 'w') as f:
        json.dump(move_to_id, f)
    logging.info(f"Vocabulary saved to {vocab_file}")

    # --- 2. Construction du label map ---
    player_names = df_raw['player'].unique().to_list()
    player_to_label = {name: i for i, name in enumerate(player_names)}
    label_to_player = {i: name for name, i in player_to_label.items()}

    with open(label_map_file, 'w') as f:
        json.dump(player_to_label, f)
    logging.info(f"Label map saved to {label_map_file}: {player_to_label}")

    # 2. Chunking and Tokenization
    processed_data = [] 

    num_moves_to_sample = SEQUENCE_LENGTH - 2 

    for row in tqdm(df_raw.rows(named=True), total=len(df_raw), desc="Sampling games"):
        moves_str = row['moves']
        player_name = row['player']
        
        color = row['color']
        color_token = '[WHITE]' if color == 'WHITE' else '[BLACK]'
        
        moves_list = moves_str.split(' ')
        label_id = player_to_label[player_name]
        n_moves = len(moves_list) # This is the area of your error

        ### --- START: RANDOM SAMPLING LOGIC --- ###
        
        if n_moves >= num_moves_to_sample:
            
            last_possible_start = n_moves - num_moves_to_sample
            start_idx = random.randint(0, last_possible_start)
            
            move_chunk = moves_list[start_idx : start_idx + num_moves_to_sample]
            
            token_list = ['[CLS]', color_token] + move_chunk
            
            token_ids = [move_to_id.get(token, 1) for token in token_list]
            
            processed_data.append({
                'token_ids': token_ids,
                'label_id': label_id
            })
            
        ### --- END: RANDOM SAMPLING LOGIC --- ###

    # 4. Save the final preprocessed dataset
    df_processed = pl.DataFrame(processed_data)

    df_processed.write_parquet(output_parquet)

    logging.info("\nPreprocessing complete.")
    logging.info(f"Original {len(df_raw)} games resulted in {len(df_processed)} high-quality sequences.")
    logging.info(f"Processed data saved to {output_parquet}.")
    logging.info("\nHead of processed DataFrame:")
    logging.info(df_processed.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize chess games.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the raw input CSV.")
    parser.add_argument('--vocab_file', type=str, required=True, help="Path to save the vocab.json.")
    parser.add_argument('--label_map_file', type=str, required=True, help="Path to save the label_map.json.")
    parser.add_argument('--output_parquet', type=str, required=True, help="Path to save the processed .parquet file.")
    
    args = parser.parse_args()
    main(args.input_csv, args.vocab_file, args.label_map_file, args.output_parquet)
