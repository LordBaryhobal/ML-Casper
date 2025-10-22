import random
from typing import Optional

import chess
from chess import Board, IllegalMoveError, Outcome
from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stockfish import Stockfish

app = FastAPI()

router = APIRouter()

stockfish = Stockfish("stockfish/stockfish-ubuntu-x86-64-avx2")


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


@router.post("/evaluate/")
def evaluate(game_info: EvaluationData):
    # TODO: Fancy ML model
    return {
        "elo": random.randint(700, 2000),
        "evaluations": [random.random() for _ in range(len(game_info.moves))]
    }


@router.post("/predict/")
def predict_move(game_info: PredictData):
    stockfish.set_fen_position(game_info.fen)
    stockfish.set_elo_rating(game_info.elo)
    move: Optional[str] = stockfish.get_best_move()
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


app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(
    directory="public",
    html=True
), name="static")
