from pathlib import Path

import polars as pl
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DIR = Path(__file__).parent
IN_PATH = DIR / "evaluated2.parquet"

print("Loading data")
df: pl.LazyFrame = pl.scan_parquet(IN_PATH)

print("Labeling players")
whites: pl.LazyFrame = df.filter(pl.col("move_idx").mod(2) == 0).rename({"white_elo": "elo"}).drop("black_elo").with_columns(player=pl.lit("white"))
blacks: pl.LazyFrame = df.filter(pl.col("move_idx").mod(2) == 1).rename({"black_elo": "elo"}).drop("white_elo").with_columns(player=pl.lit("black"))

print("Computing features")
all_moves = pl.concat([whites, blacks]).collect().to_pandas()

all_moves = all_moves.sort_values(['game_id', 'player', 'move_idx'])
grouped = all_moves.groupby(['game_id', 'player'])

print(" - mean_cp_loss")
all_moves['mean_cp_loss'] = grouped['cp_loss'].expanding().mean().reset_index(level=[0,1], drop=True)

print(" - mean_best_diff")
all_moves['mean_best_diff'] = grouped['best_diff'].expanding().mean().reset_index(level=[0,1], drop=True)

print(" - std_cp_loss")
all_moves['std_cp_loss'] = grouped['cp_loss'].expanding().std().reset_index(level=[0,1], drop=True)

all_moves['move_count'] = all_moves['move_idx']

# Drop rows with NaN (early moves with std undefined)
all_moves = all_moves.dropna(subset=['std_cp_loss'])

print("Splitting train/test")
features = ['move_count', 'mean_cp_loss', 'mean_best_diff', 'std_cp_loss']
target = 'elo'

X = all_moves[features]
y = all_moves[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training")
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.1f} ELO points")