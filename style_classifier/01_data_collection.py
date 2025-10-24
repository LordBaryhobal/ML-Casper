# src/01_data_collection.py
import requests
import polars as pl
import chess.pgn
import io
from tqdm import tqdm
import argparse  
import logging   
import os       
import config as cfg


def fetch_player_games(username, num_games):
    """
    Fetches and correctly parses a stream of PGN games for a Lichess user.
    """
    api_url = f"https://lichess.org/api/games/user/{username}"
    params = {'max': num_games, 'rated': 'true', 'perfType': 'bullet,blitz,rapid,classical'}
    headers = {'Accept': 'application/x-pgn'}
    
    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        
        # Lichess separates games with two newlines. We split the whole text response.
        pgn_texts = response.text.strip().split('\n\n\n')
        
        parsed_games = []
        for pgn_string in pgn_texts:
            if pgn_string:
                game = chess.pgn.read_game(io.StringIO(pgn_string))
                if game:
                    parsed_games.append(game)
        return parsed_games

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data for {username}: {e}")
        return []
    
def main(output_file):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Démarrage de la collecte de données...")

    all_player_games = []

    for username, label_name in tqdm(cfg.PLAYER_CONFIG.items(), desc="Processing Players"):
        logging.info(f"Fetching games for {label_name} ({username})...")
        games = fetch_player_games(username, cfg.TOTAL_GAMES_TO_FETCH)

        if not games:
            logging.warning(f"Could not retrieve games for {label_name}. Skipping.")
            continue
        
        white_games = []
        black_games = []

        logging.info(f"Retrieved {len(games)} games. Balancing roles and extracting moves...")

        for game in games:
            if len(white_games) >= cfg.GAMES_PER_ROLE and len(black_games) >= cfg.GAMES_PER_ROLE:
                break

            white_player = game.headers.get("White", "").lower()
            black_player = game.headers.get("Black", "").lower()
            
            # This correctly sets up the board from the game's actual starting position (including FENs)
            board = game.headers.board()
            moves_list = []
            for move in game.mainline_moves():
                moves_list.append(board.san(move))
                board.push(move)
            moves_string = " ".join(moves_list)

            # --- FINAL FIX: Only add the game if we successfully extracted moves ---
            if moves_string:
                if username.lower() in white_player and len(white_games) < cfg.GAMES_PER_ROLE:
                    white_games.append({'player': label_name, 'color': 'WHITE', 'moves': moves_string})
                elif username.lower() in black_player and len(black_games) < cfg.GAMES_PER_ROLE:
                    black_games.append({'player': label_name, 'color': 'BLACK', 'moves': moves_string})

        logging.info(f"Collected {len(white_games)} games as White and {len(black_games)} as Black.")
        all_player_games.extend(white_games)
        all_player_games.extend(black_games)

    logging.info(f"data collection complete. Total games collected: {len(all_player_games)}")

    df = pl.DataFrame(all_player_games)
    
    if not df.is_empty():
        logging.info(f"Successfully created a DataFrame with {len(df)} total games.")
        
        # S'assurer que le répertoire de sortie existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.write_csv(output_file)
        logging.info(f"Data saved successfully to {output_file}")
        logging.info(f"DataFrame head:\n{df.head()}")
    else:
        logging.warning("No data was collected. The DataFrame is empty.")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect chess games from Lichess API.")
    parser.add_argument(
        '--output_csv', 
        type=str, 
        required=True, 
        help="Path to save the output CSV file (e.g., data/raw/player_games_from_api.csv)"
    )
    args = parser.parse_args()
    main(args.output_csv)
