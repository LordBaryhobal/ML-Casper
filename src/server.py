from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

router = APIRouter()

app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(
    directory="public",
    html=True
), name="static")
