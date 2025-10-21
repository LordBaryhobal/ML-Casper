import uvicorn

from src.server import app


def main():
    uvicorn.run(app, host="0.0.0.0", port=35432)


if __name__ == "__main__":
    main()
