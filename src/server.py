import os
import random
from typing import Optional
import pandas as pd
from xgboost import XGBRegressor
import json
import numpy as np
import threading

import chess
from chess import Board, IllegalMoveError, Move, Outcome
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.style.model import StyleAnalyser
from stockfish import Stockfish

load_dotenv()


app = FastAPI()

router = APIRouter()


stockfish_path: str = os.environ["STOCKFISH_PATH"]
print(f"Stockfish path: {stockfish_path}")

sf_move = Stockfish(stockfish_path)
sf_move.update_engine_parameters({"UCI_LimitStrength": "true", "Hash": 1024})
sf_eval = Stockfish(stockfish_path)
sf_eval.update_engine_parameters({"Hash": 1024})

SF_MOVE_LOCK = threading.Lock()
SF_EVAL_LOCK = threading.Lock()

MODEL = None
MODEL_FEATURES: list[str] = []

def load_model():
    global MODEL, MODEL_FEATURES
    try:
        with open("models/elo_xgb.meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            MODEL_FEATURES = meta["features"]

        mdl = XGBRegressor()
        mdl.load_model("models/elo_xgb.json")
        MODEL = mdl
        print(f"✔ Model loaded with features: {MODEL_FEATURES}")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        MODEL = None
        MODEL_FEATURES = []

load_model()

def cp_loss_series(moves_uci: list[str]) -> list[dict]:
    rows = []
    board = chess.Board()
    with SF_EVAL_LOCK:
        sf_eval.set_fen_position(board.fen())
        current_eval: float = sf_eval.get_evaluation()["value"]
        for i, uci in enumerate(moves_uci):
            best: dict = sf_eval.get_top_moves(1)[0]
            best_eval: float = best["Centipawn"]

            sf_eval.make_moves_from_current_position([uci])
            new_eval: float = sf_eval.get_evaluation()["value"]

            cp_loss: float = new_eval - current_eval
            current_eval = new_eval
            if best["Mate"]:
                continue
            best_diff: float = best_eval - new_eval
            
            rows.append({
                "move_idx": i,
                "cp_loss": cp_loss,
                "best_diff": best_diff
            })
    return rows

def predict_elo_from_features(df: pd.DataFrame) -> np.float32:
    if MODEL is None:
        raise RuntimeError("Model not initialized")
    X = df[["move_count", "mean_cp_loss", "mean_best_diff", "std_cp_loss"]]
    preds = MODEL.predict(X.iloc[-1:])
    return np.mean(preds)

style_analyser: StyleAnalyser = StyleAnalyser()


class MoveCheckData(BaseModel):
    fen: str
    move: str


class EvaluationData(BaseModel):
    player: str
    moves: list[str]


class PredictData(BaseModel):
    fen: str
    elo: int


class GameOutcome(BaseModel):
    termination: str
    result: str
    winner: Optional[str]

    @staticmethod
    def make(outcome: Outcome):
        return GameOutcome(
            termination=outcome.termination.name,
            result=outcome.result(),
            winner="white" if outcome.winner == chess.WHITE else "black"
        )


@router.post("/move/")
def apply_move(move: MoveCheckData):
    board: Board = Board(move.fen)
    try:
        board.push_uci(move.move)
        outcome: Optional[Outcome] = board.outcome()
        return {
            "valid": True,
            "fen": board.fen(),
            "outcome": GameOutcome.make(outcome) if outcome is not None else None
        }
    except IllegalMoveError:
        return {
            "valid": False
        }

# TODO check if mean all moves or just current player
@router.post("/evaluate/")
def evaluate(game_info: EvaluationData):
    moves_uci: list[str] = game_info.moves or []
    move_count: int = len(moves_uci)

    df: pd.DataFrame = pd.DataFrame(cp_loss_series(moves_uci))

    df["mean_cp_loss"] = df["cp_loss"].expanding().mean().reset_index(drop=True)
    df["mean_best_diff"] = df["best_diff"].expanding().mean().reset_index(drop=True)
    df["std_cp_loss"] = df["cp_loss"].expanding().std().reset_index(drop=True)
    df["move_count"] = df["move_idx"]

    if game_info.player == "white":
        mod = 0
    else:
        mod = 1
    
    df = df.iloc[mod::2]

    # Drop rows with NaN (early moves with std undefined)
    df = df.dropna(subset=["std_cp_loss"])

    pred = predict_elo_from_features(df)

    if pred is None or not np.isfinite(pred):
        print("Could not predict ELO, returning random value")
        pred = float(random.randint(700, 2000))

    return {
        "elo": round(float(pred))
    }

@router.post("/predict/")
def predict_move(game_info: PredictData):
    with SF_MOVE_LOCK:
        sf_move.set_fen_position(game_info.fen)
        sf_move.set_elo_rating(game_info.elo)
        move: Optional[str] = sf_move.get_best_move()
    if move is None:
        return {
            "error": "Already mate"
        }
    board: Board = Board(game_info.fen)
    board.push_uci(move)
    outcome: Optional[Outcome] = board.outcome()
    return {
        "move": move,
        "fen": board.fen(),
        "outcome": GameOutcome.make(outcome) if outcome is not None else None
    }


@router.post("/analyse/")
def analyse(game_info: EvaluationData):
    moves: list[Move] = [
        Move.from_uci(m)
        for m in game_info.moves
    ]
    board: Board = Board()
    san_moves: list[str] = []
    for m in moves:
        san_moves.append(board.san(m))
        board.push(m)
    game_data: tuple[str, str] = (" ".join(san_moves), game_info.player.upper())
    style_probs: dict[str, float] = style_analyser.infer(game_data)
    return {
        "probabilities": style_probs
    }


app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(
    directory="public",
    html=True
), name="static")
