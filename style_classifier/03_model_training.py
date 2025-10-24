# src/03_model_training.py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import json
import os
import argparse  
import logging  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_model import ChessGameDataset, StyleTransformer
import config as cfg

# --- NEW IMPORTS FOR RAY TUNE ---
import tempfile
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from functools import partial
# ----------------------------------


def load_artifacts(vocab_file, label_map_file, data_file):
    logging.info("Loading preprocessing artifacts...")
    try:
        with open(vocab_file, 'r') as f:
            move_to_id = json.load(f)
        VOCAB_SIZE = len(move_to_id)
        logging.info(f"Loaded vocabulary. Vocab size: {VOCAB_SIZE}")

        with open(label_map_file, 'r') as f:
            player_to_label = json.load(f)
        NUM_CLASSES = len(player_to_label)
        label_to_player = {v: k for k, v in player_to_label.items()}
        logging.info(f"Loaded label map. Num classes: {NUM_CLASSES}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"{data_file} not found.")
        
        return VOCAB_SIZE, NUM_CLASSES, label_to_player
    except FileNotFoundError as e:
        logging.error(f"Error loading artifacts: {e}")
        logging.error("Please run 02_preprocessing.py first.")
        exit(1)

def setup_dataloaders(data_file, test_split_size, batch_size):
    # This function is now called by the trainable, using the tuned batch_size
    logging.info("Setting up DataLoaders...")
    dataset = ChessGameDataset(data_file)
    logging.info(f"Total sequences in dataset: {len(dataset)}")
    
    dataset_indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        dataset_indices,
        test_size=test_split_size,
        random_state=42,
        stratify=dataset.labels
    )
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    logging.info(f"Training sequences: {len(train_subset)}")
    logging.info(f"Validation sequences: {len(val_subset)}")
    return train_loader, val_loader

def plot_history(history, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss over Epochs')

    # Plot Accuracy
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Validation Accuracy over Epochs')

    plt.tight_layout()
    plt.savefig(output_path)

def plot_confusion_matrix(all_labels, all_predictions, label_names, output_path):
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)

# --- NEW: RAY TUNE TRAINABLE FUNCTION ---
def train_model_tune(config, args):
    """
    Ray Tune compatible trainable function.
    'config' contains hyperparameters to tune.
    'args' contains fixed file paths and other static configs.
    """
    
    # --- 1. Configuration & Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Starting trial on device: {device}")

    # --- 2. Load Artifacts (from fixed args) ---
    VOCAB_SIZE, NUM_CLASSES, label_to_player = load_artifacts(
        args.vocab_file, args.label_map_file, args.processed_data_file
    )

    # --- 3. Setup DataLoaders (using tuned config) ---
    train_loader, val_loader = setup_dataloaders(
        args.processed_data_file, 
        args.test_split_size,       # Fixed arg
        config["batch_size"]        # Tuned config
    )

    # --- 4. Instantiate Model (using tuned config) ---
    model = StyleTransformer(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        d_model=config["d_model"],
        nhead=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        seq_len=args.sequence_length # Fixed arg
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5
    )

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    # Use fixed num_epochs as the max, Ray's scheduler can stop trials early
    for epoch in range(args.num_epochs):
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]", leave=False)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_accuracy = 100 * correct_predictions / total_predictions
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        
        scheduler.step(epoch_val_loss)
        
        logging.info(f"Epoch {epoch+1}/{args.num_epochs} | "
                     f"Train Loss: {epoch_train_loss:.4f} | "
                     f"Val Loss: {epoch_val_loss:.4f} | "
                     f"Val Acc: {epoch_val_accuracy:.2f}%")

        # --- MODIFICATION: Checkpointing and Reporting to Ray ---
        checkpoint = None
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            
            # Create a checkpoint in a temporary directory
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint_path = os.path.join(temp_checkpoint_dir, "model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                
                # Create the checkpoint object pointing to the temp dir
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                logging.info(f"  -> New best model checkpoint saved with Val Loss: {best_val_loss:.4f}")
                
                # Report metrics AND the new checkpoint
                # This MUST be inside the 'with' block
                ray.train.report(
                    {"val_loss": epoch_val_loss, "val_accuracy": epoch_val_accuracy},
                    checkpoint=checkpoint
                )
        
        else:
            epochs_no_improve += 1
            logging.warning(f"  -> Validation loss did not improve for {epochs_no_improve} epoch(s).")

            # Report metrics *without* a checkpoint
            ray.train.report(
                {"val_loss": epoch_val_loss, "val_accuracy": epoch_val_accuracy},
                checkpoint=None
            )
        
        # Check for early stopping *after* reporting

        if epochs_no_improve >= args.early_stopping_patience:
            logging.info(f"--- Early stopping triggered after {epoch+1} epochs. ---")
            break # Exit the training loop for this trial
            
    logging.info("Trial training complete.")


# --- NEW: MAIN FUNCTION TO LAUNCH TUNING ---
def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # S'assurer que les répertoires de sortie (pour les rapports finaux) existent
    os.makedirs(os.path.dirname(args.history_plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.cm_plot_path), exist_ok=True)

    # --- 1. Define Search Space ---
    param_space = {
        # Optimizer params from config.py
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64]),

        # Model architecture params from config.py
        "d_model": tune.choice([512, 768]),
        "num_heads": tune.choice([8, 16]),
        "num_encoder_layers": tune.choice([6, 8, 12]),
        "dim_feedforward": tune.choice([3072, 4096]),
        "dropout": tune.uniform(0.0, 0.2)
    }

    # --- 2. Configure Scheduler and Search Algorithm ---
    # ASHA Scheduler stops unpromising trials early
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=args.num_epochs, # Max epochs per trial
        grace_period=5,        # Min epochs before stopping a trial
        reduction_factor=2     # Halve the number of trials every 'grace_period'
    )

    # Use Optuna for intelligent search (requires pip install optuna)
    search_alg = OptunaSearch(
        metric="val_loss",
        mode="min"
    )

    # --- 3. Setup the Trainable (binding fixed args) ---
    # We use partial to "freeze" the 'args' namespace as a fixed parameter
    # for our trainable function.
    trainable_with_args = partial(train_model_tune, args=args)
    
    # --- 4. Configure and Run the Tuner ---
    ray.init()

    # Configure resource allocation (optional, but good practice)
    # This allocates 1 GPU per trial. Adjust as needed.
    resources_per_trial = {"cpu": 32, "gpu": 1 if torch.cuda.is_available() else 0}
    
    tuner = tune.Tuner(
        tune.with_resources(trainable_with_args, resources_per_trial),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=50,  # Number of different hyperparameter configs to try
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=train.RunConfig(
            name="chess_style_transformer_tune",
            storage_path="~/ray_results", # Directory to save all trial logs
            # This tells Ray to save the best checkpoint based on validation loss
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
    )

    results = tuner.fit() # This runs the full tuning job

    # --- 5. Get Best Results and Run Final Evaluation ---
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    logging.info("--- Hyperparameter Tuning Complete ---")
    logging.info(f"Best validation loss: {best_result.metrics['val_loss']:.4f}")
    logging.info(f"Best validation accuracy: {best_result.metrics['val_accuracy']:.2f}%")
    logging.info(f"Best hyperparameters found: \n{json.dumps(best_result.config, indent=2)}")
    
    best_checkpoint_path = os.path.join(best_result.checkpoint.path, "model.pth")
    logging.info(f"Best model saved in: {best_checkpoint_path}")

    # --- 6. Final Evaluation & Plotting using the BEST model ---
    logging.info("Running final evaluation on the BEST model...")
    
    # Load artifacts and data
    VOCAB_SIZE, NUM_CLASSES, label_to_player = load_artifacts(
        args.vocab_file, args.label_map_file, args.processed_data_file
    )
    # Use best batch_size for final eval
    best_batch_size = best_result.config["batch_size"]
    _, val_loader = setup_dataloaders(
        args.processed_data_file, args.test_split_size, best_batch_size
    )

    # Instantiate the best model
    best_config = best_result.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StyleTransformer(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        d_model=best_config["d_model"],
        nhead=best_config["num_heads"],
        num_encoder_layers=best_config["num_encoder_layers"],
        dim_feedforward=best_config["dim_feedforward"],
        dropout=best_config["dropout"],
        seq_len=args.sequence_length
    ).to(device)
    
    # Load the best checkpoint
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    model.eval()

    # Run evaluation
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Final Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    player_names = [label_to_player[i] for i in range(NUM_CLASSES)]
    logging.info("\n--- Final Classification Report (Best Model) ---")
    logging.info("\n" + classification_report(all_labels, all_predictions, target_names=player_names))
    
    plot_confusion_matrix(all_labels, all_predictions, player_names, args.cm_plot_path)
    
    # Note: plot_history is less relevant as Ray Tune provides TensorBoard logging.
    # We plot the confusion matrix for the best model.
    logging.info(f"Final confusion matrix saved to {args.cm_plot_path}")
    logging.info("--- 03_model_training.py (with Ray Tune) Complete ---")


# --- MODIFIED: __main__ block ---
if __name__ == "__main__":
    # This block now ONLY parses FIXED arguments.
    # Hyperparameters are defined in the 'main' function's 'param_space'.
    
    parser = argparse.ArgumentParser(description="Tune a transformer model for chess style classification.")
    
    # --- Fichiers (FIXED) ---
    parser.add_argument('--processed_data_file', type=str, default="data/processed/preprocessed_games.parquet")
    parser.add_argument('--vocab_file', type=str, default="models/vocab.json")
    parser.add_argument('--label_map_file', type=str, default="models/label_map.json")
    
    # --- Rapports (FIXED) ---
    parser.add_argument('--history_plot_path', type=str, default="reports/training_history.png")
    parser.add_argument('--cm_plot_path', type=str, default="reports/best_model_confusion_matrix.png")

    # --- Paramètres d'entraînement (FIXED) ---
    # These are now the *maximums* or constants for the tuning run.
    parser.add_argument('--num_epochs', type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument('--test_split_size', type=float, default=cfg.TEST_SPLIT_SIZE)
    parser.add_argument('--early_stopping_patience', type=int, default=cfg.EARLY_STOPPING_PATIENCE)

    # --- Paramètres du modèle (FIXED) ---
    parser.add_argument('--sequence_length', type=int, default=cfg.SEQUENCE_LENGTH)

    # Note: All other hyperparameters (lr, batch_size, d_model, etc.)
    # are removed from argparse and defined in 'param_space' in main().

    args = parser.parse_args()
    # --- THIS IS THE FIX ---
    # Convert all relative paths to absolute paths BEFORE passing them to Ray
    args.processed_data_file = os.path.abspath(args.processed_data_file)
    args.vocab_file = os.path.abspath(args.vocab_file)
    args.label_map_file = os.path.abspath(args.label_map_file)
    args.history_plot_path = os.path.abspath(args.history_plot_path)
    args.cm_plot_path = os.path.abspath(args.cm_plot_path)
    main(args)
