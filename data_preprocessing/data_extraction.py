import chess.pgn
import csv
from pathlib import Path
from chess.pgn import StringExporter


file_path = Path(__file__).parent / "data/lichess_db_standard_rated_2015-03.pgn"
output = file_path.with_suffix(".csv")

def winner_from_result(result: str):
    if result == "1-0":
        return "White"
    elif result == "0-1":
        return "Black"
    elif result in ("1/2-1/2", "½-½", "0.5-0.5"):
        return "Draw"
    return ""

with open(file_path, encoding="utf-8", errors="ignore") as pgn, open(output, "w", newline="", encoding="utf-8") as csv_file:
    columns = ["Event", "Site", "Date", "Round", "White", "Black", "Result", "UTCDate", "UTCTime", "WhiteElo", "BlackElo", 
                "WhiteRatingDiff", "BlackRatingDiff", "ECO", "Opening", "TimeControl", "Termination", "LichessId", 
                "Winner", "Moves"]

    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()

    pgn.seek(0)

    count = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        headers = game.headers
        row = {key: headers[key] for key in columns if key in headers}
        ply_count = game.end().board().ply()
        row["Winner"] = winner_from_result(row.get("Result", ""))
        exporter = StringExporter(columns=None, headers=False, variations=False, comments=False)
        row["Moves"] = game.accept(exporter).strip()

        writer.writerow(row)

        count += 1
        if count % 1000 == 0:
            print(f"{count} converted game...")

    print(f"Convert finish, CSV file : {output}")
