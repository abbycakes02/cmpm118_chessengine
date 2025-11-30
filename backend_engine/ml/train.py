import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import time
import os
import csv

from .model import ChessValueNet
from .data_loader import get_train_test_loaders

# --- Config ---
# double dirname to get to the backend_engine directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

# Training Hyperparameters
BATCH_SIZE = 2048
LEARNING_RATE = 0.002
EPOCHS = 3
VALIDATION_SPLIT = 0.1
NUM_WORKERS = 8

# achitecture Hyperparameters
NUM_CHANNELS = 32
NUM_RESIDUAL_BLOCKS = 3
# --------------


def train():
    # check if GPU is available
    use_amp = False
    amp_device = 'cpu'
    pin_memory = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        amp_device = 'cuda'
        pin_memory = True
        print(f"GPU detected, using device: {device} with AMP and Pin memory enabled.")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False
        amp_device = 'cpu'
        pin_memory = False
        print(f"MPS device detected, using device: {device} with AMP and Pin memory disabled.")

    else:
        device = torch.device("cpu")
        amp_device = 'cpu'
        use_amp = False
        pin_memory = False
        print(f"Using CPU device: {device} without AMP or Pin memory.")

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
        writer.writerow(["timestamp", "epoch", "chunk", "train_loss", "val_loss"])

    # Initialize the model, loss function, and optimizer
    model = ChessValueNet(
        num_channels=NUM_CHANNELS,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS
        ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # set up AMP if running on gpu
    scaler = GradScaler(enabled=use_amp)

    start_time = time.time()
    print("\n Starting Training With hyperparameters:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Validation Split: {VALIDATION_SPLIT}")
    print(f"  Number of Channels: {NUM_CHANNELS}")
    print(f"  Number of Residual Blocks: {NUM_RESIDUAL_BLOCKS}")
    print(f"  Using AMP: {use_amp}")
    print(f"  Start Time: {time.ctime(start_time)}")

    for epoch in range(EPOCHS):
        print(f"\n --- Epoch {epoch + 1}/{EPOCHS} ---")

        # get a loader for each chunk of data
        train_loader, test_loader, n_train, n_test = get_train_test_loaders(
            DATA_DIR,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory
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

                # check if using AMP, then use autocast to speed up training by using mixed precision where possible
                with autocast(enabled=use_amp, device_type=amp_device, dtype=torch.float16):
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Backward pass and optimization with AMP scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                chunk_loss += loss.item()
                steps += 1

            avg_loss = chunk_loss / steps if steps > 0 else 0
            epoch_train_loss += avg_loss
            train_chunks += 1
            print(f'    [Chunk {i + 1}] Training Loss: {avg_loss:.6f}, Elapsed Time: {time.time() - start_time:.2f} seconds')
            with open(log_path, "a") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([f"{time.time() - start_time:.2f}", epoch + 1, i + 1, f"{avg_loss:.6f}", ""])

        avg_epoch_train_loss = epoch_train_loss / train_chunks if train_chunks > 0 else 0
        print(f"  Average Training Loss for Epoch {epoch + 1}: {avg_epoch_train_loss:.6f}, Elapsed Time: {time.time() - start_time:.2f} seconds")

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
        print(f"  Average Validation Loss for Epoch {epoch + 1}: {avg_epoch_val_loss:.6f}, in {time.time() - start_time:.2f} seconds")
        with open(log_path, "a") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([f"{time.time() - start_time:.2f}", epoch + 1, "end", "", f"{avg_epoch_val_loss:.6f}"])

        # after validation, save model checkpoint and log results
        checkpoint_path = os.path.join(session_model_dir, f"epoch_{epoch + 1}_{NUM_CHANNELS}ch_{NUM_RESIDUAL_BLOCKS}resblocks.pth")
        torch.save(model.state_dict(), checkpoint_path)

    model_filename = (f"chess_valuenet_{NUM_CHANNELS}ch_{NUM_RESIDUAL_BLOCKS}resblocks.pth")
    final_model_path = os.path.join(session_model_dir, model_filename)
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete in {time.time() - start_time:.2f} seconds.")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    train()
