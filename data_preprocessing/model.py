import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print(f"Using {device.type}")

model = nn.Sequential(
    nn.Linear(65, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
).to(device)

loss_fit = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

print("Loading dataframe")
df: pl.DataFrame = pl.read_parquet("output/per_move_sampled.parquet")
unique_ids = df.select("id").unique().to_series().to_list()

rng = np.random.default_rng(seed=42)
rng.shuffle(unique_ids)

split_idx = int(len(unique_ids) * 0.7)
train_ids = set(unique_ids[:split_idx])
test_ids  = set(unique_ids[split_idx:])

print(f"Total games: {len(unique_ids)} → train={len(train_ids)}, test={len(test_ids)}")

df_train = df.filter(pl.col("id").is_in(train_ids))
df_test  = df.filter(pl.col("id").is_in(test_ids))

print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

print("Splitting features and labels")
X_train = df_train.select("move_idx", *[f"tile{i}{j}" for i in range(8) for j in range(8)]).to_torch(dtype=pl.Float32)
y_train = df_train.select("white_elo", "black_elo").to_torch(dtype=pl.Float32)

X_test = df_test.select("move_idx", *[f"tile{i}{j}" for i in range(8) for j in range(8)]).to_torch(dtype=pl.Float32)
y_test = df_test.select("white_elo", "black_elo").to_torch(dtype=pl.Float32)

n_epochs = 100
batch_size = 10000

train_loss = []
test_loss = []

print("Training")
os.makedirs("learning_curves", exist_ok=True)
for epoch in tqdm.tqdm(range(n_epochs)):
    batch_loss = []
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_pred = model(X_batch)
        y_batch = y_train[i:i+batch_size]
        loss = loss_fit(y_pred, y_batch)
        batch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    mean_batch_loss = np.array(batch_loss).mean()
    train_loss.append(mean_batch_loss)
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss.append(loss_fit(y_pred, y_test).item())
    print(f"Finished epoch {epoch}, latest loss: train={mean_batch_loss} / test={test_loss[-1]}")

    epochs = list(range(epoch + 1))
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, test_loss, label="Test")
    plt.savefig(f"learning_curves/epoch_{epoch}.png")

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X_test)
accuracy = (y_pred.round() == y_test).float().mean()
print(f"Accuracy {accuracy}")

torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}, "model.pt")
