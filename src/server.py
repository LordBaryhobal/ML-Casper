import os
import random
from typing import Optional
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
sf_eval = Stockfish(stockfish_path)

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

#TODO adapt
def eval_cp(ev: dict) -> int:
    if ev["type"] == "cp":
        return int(ev["value"])
    return 32000 if ev["value"] > 0 else -32000

def cp_loss_series(moves_uci: list[str]) -> list[int]:
    diffs = []
    board = chess.Board()
    for uci in moves_uci:
        # --- NEW: on utilise sf_eval + lock
        with SF_EVAL_LOCK:
            sf_eval.set_fen_position(board.fen())
            best = sf_eval.get_best_move()

            if best:
                sf_eval.set_fen_position(board.fen())
                best_eval = sf_eval.get_evaluation()
            else:
                best_eval = {"type": "cp", "value": 0}

        board.push_uci(uci)

        with SF_EVAL_LOCK:
            sf_eval.set_fen_position(board.fen())
            played_eval = sf_eval.get_evaluation()

        diffs.append(abs(eval_cp(best_eval) - eval_cp(played_eval)))
    return diffs

def predict_elo_from_features(move_count: int, mean_cp_loss: float, mean_best_diff: float, std_cp_loss: float) -> float | None:
    if MODEL is None:
        return None
    feats = {
        "move_count": float(move_count),
        "mean_cp_loss": float(mean_cp_loss),
        "mean_best_diff": float(mean_best_diff),
        "std_cp_loss": float(std_cp_loss),
    }
    x = np.array([[feats[name] for name in MODEL_FEATURES]], dtype=np.float32)
    pred = MODEL.predict(x)[0]
    return float(pred)

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
    moves_uci = game_info.moves or []
    move_count = len(moves_uci)

    losses = cp_loss_series(moves_uci) if move_count else []

    if losses:
        mean_cp = float(np.mean(losses))
        std_cp = float(np.std(losses)) if len(losses) > 1 else 0.0
    else:
        mean_cp = 0.0
        std_cp = 0.0

    mean_best_diff = mean_cp

    pred = predict_elo_from_features(
        move_count=int(move_count),
        mean_cp_loss=mean_cp,
        mean_best_diff=mean_best_diff,
        std_cp_loss=std_cp
    )

    if pred is None or not np.isfinite(pred):
        pred = float(random.randint(700, 2000))

    return {
        "elo": round(float(pred)),
        "evaluations": losses,
        "features": {
            "move_count": move_count,
            "mean_cp_loss": mean_cp,
            "mean_best_diff": mean_best_diff,
            "std_cp_loss": std_cp
        }
    }

@router.post("/predict/")
def predict_move(game_info: PredictData):
    # --- NEW: use sf_move + lock
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
