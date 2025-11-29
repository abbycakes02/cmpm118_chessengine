import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import csv

from model import ChessValueNet
from data_loader import get_train_test_loaders

# --- Config ---
# double dirname to get to the backend_engine directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 5
VALIDATION_SPLIT = 0.1
# --------------


def train():
    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model directory: {MODEL_DIR}")

    # create new model save directory for this training session
    session = f"session_{int(time.time())}"
    session_model_dir = os.path.join(MODEL_DIR, session)
    os.makedirs(session_model_dir)
    print(f"Created session model directory: {session_model_dir}")

    # log train and valiation loss to a csv for analysis later
    log_path = os.path.join(session_model_dir, "training_log.csv")
    with open(log_path, "w") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "time_elapsed"])

    # Initialize the model, loss function, and optimizer
    model = ChessValueNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n --- Epoch {epoch + 1}/{EPOCHS} ---")

        # get a loader for each chunk of data
        train_loader, test_loader, n_train, n_test = get_train_test_loaders(
            DATA_DIR,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            num_workers=4
        )

        print(f"  Training on {n_train} files")
        # put the model in training mode
        model.train()
        epoch_train_loss = 0.0
        train_chunks = 0

        for i, loader in enumerate(train_loader):
            chunk_loss = 0.0
            steps = 0

            for inputs, targets in loader:
                # move tensors to the configured device
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                chunk_loss += loss.item()
                steps += 1

            avg_loss = chunk_loss / steps if steps > 0 else 0
            epoch_train_loss += avg_loss
            train_chunks += 1
            print(f'    [Chunk {i + 1}] Training Loss: {avg_loss:.6f}')

        avg_epoch_train_loss = epoch_train_loss / train_chunks if train_chunks > 0 else 0
        print(f"  Average Training Loss for Epoch {epoch + 1}: {avg_epoch_train_loss:.6f}")

        print(f"  Validating on {n_test} files")
        # put the model in evaluation mode to evaluate on validation set
        model.eval()
        epoch_val_loss = 0.0
        val_chunks = 0

        with torch.no_grad():
            for loader in test_loader:
                chunk_loss = 0.0
                steps = 0

                for inputs, targets in loader:
                    # move tensors to the configured device
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    chunk_loss += loss.item()
                    steps += 1

                avg_loss = chunk_loss / steps if steps > 0 else 0
                epoch_val_loss += avg_loss
                val_chunks += 1

        avg_epoch_val_loss = epoch_val_loss / val_chunks if val_chunks > 0 else 0
        print(f"  Average Validation Loss for Epoch {epoch + 1}: {avg_epoch_val_loss:.6f}")

        # after validation, save model checkpoint and log results
        checkpoint_path = os.path.join(session_model_dir, f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        with open(log_path, "a") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                epoch + 1,
                f"{avg_epoch_train_loss:.6f}",
                f"{avg_epoch_val_loss:.6f}",
                f"{time.time() - start_time:.2f}"
            ])

    final_model_path = os.path.join(session_model_dir, "chess_value_net_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete in {time.time() - start_time:.2f} seconds.")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    train()
