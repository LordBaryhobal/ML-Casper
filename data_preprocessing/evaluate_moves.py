import io
import os
from pathlib import Path
from typing import Optional
import chess
import chess.pgn
import dotenv
import multiprocessing as mp
import polars as pl
from pyarrow import Table
import pyarrow.parquet as pq
from stockfish import Stockfish
import tqdm

dotenv.load_dotenv()

#stockfish = Stockfish(os.environ["STOCKFISH_PATH"])
#stockfish.update_engine_parameters({"Threads": 3})
#stockfish.set_depth(2)

n_workers = 8

DIR = Path(__file__).parent
IN_PATH = DIR / "output.parquet"
OUT_PATH = DIR / "evaluated.parquet"

def evaluate_game(row: dict, stockfish: Stockfish):
    game_id: int = row["idx"]
    white_elo: int = row["WhiteElo"]
    black_elo: int = row["BlackElo"]
    game = chess.pgn.read_game(io.StringIO(row["Moves"]))
    if game is None:
        return pl.DataFrame()
    board = game.board()
    stockfish.set_fen_position(board.fen())
    rows = []
    evaluation: float = stockfish.get_evaluation()["value"]
    for i, mv in tqdm.tqdm(enumerate(game.mainline_moves()), desc="Move"):
        top_move: dict = stockfish.get_top_moves(1)[0]
        stockfish.make_moves_from_current_position([mv.uci()])
        new_evaluation: float = stockfish.get_evaluation()["value"]
        cp_loss: float = new_evaluation - evaluation
        evaluation = new_evaluation
        if top_move["Mate"]:
            continue
        best_diff: float = top_move["Centipawn"] - new_evaluation
        
        rows.append({
            "game_id": game_id,
            "move_idx": i,
            "cp_loss": cp_loss,
            "best_diff": best_diff,
            "white_elo": white_elo,
            "black_elo": black_elo,
        })
    return pl.DataFrame(rows)

def evaluate_game_batch_chunk(df: pl.DataFrame):
    stockfish = Stockfish(os.environ["STOCKFISH_PATH"])
    stockfish.set_depth(2)

    result: list[pl.DataFrame] = []
    for game in tqdm.tqdm(df.rows(named=True), desc="Game"):
        moves: pl.DataFrame = evaluate_game(game, stockfish)
        result.append(moves)
    
    print("Finished batch chunk")
    return pl.concat(result)

def evaluate_game_batch(df: pl.DataFrame):
    chunk_size: int = len(df) // n_workers
    chunks = [
        df[c:c+chunk_size]
        for c in range(0, len(df), chunk_size)
    ]
    
    with mp.Pool(n_workers) as pool:
        results: list[pl.DataFrame] = pool.map(evaluate_game_batch_chunk, chunks)
    
    print("Finished batch")
    return pl.concat(results)

def evaluate_games(df: pl.LazyFrame):
    df = df.select("Moves", "WhiteElo", "BlackElo").with_row_index("idx")
    writer: Optional[pq.ParquetWriter] = None

    for batch in tqdm.tqdm(df.collect_batches(), desc="Batch"):
        evaluated: pl.DataFrame = evaluate_game_batch(batch)

        table: Table = evaluated.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(OUT_PATH, table.schema)
        writer.write_table(table)

df: pl.LazyFrame = pl.scan_parquet(IN_PATH).head(100)
evaluate_games(df)