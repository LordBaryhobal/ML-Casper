from chess import Board, IllegalMoveError
from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

router = APIRouter()


class MoveCheckData(BaseModel):
    fen: str
    move: str


@router.post("/move/")
def apply_move(move: MoveCheckData):
    board: Board = Board(move.fen)
    try:
        board.push_uci(move.move)
        return {
            "valid": True,
            "fen": board.fen()
        }
    except IllegalMoveError:
        return {
            "valid": False
        }


app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(
    directory="public",
    html=True
), name="static")
