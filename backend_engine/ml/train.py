import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
import time
import glob
import os
import csv
import argparse

from .model import ChessNet
from .data_loader import get_train_test_loaders
from .vocab import VOCAB_SIZE

# --- Config ---
# double dirname to get to the backend_engine directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# one more time to get to the project root
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

# Training Hyperparameters
BATCH_SIZE = 1024
MAX_LR = 0.005
EPOCHS = 5
VALIDATION_SPLIT = 0.1
NUM_WORKERS = 4

# achitecture Hyperparameters
HISTORY_LENGTH = 5  # number of past positions to include in input tensor
BOARD_CHANNELS = 20  # number of channels to represent the board state
NUM_CHANNELS = 64  # number of channels in the model hidden layers
NUM_RESIDUAL_BLOCKS = 3

# Resume training from a checkpoint (if False, starts fresh)
RESUME_TRAINING = False
RESUME_MODEL_PATH = None  # os.path.join(MODEL_DIR, "session_1764532867", "epoch_1_64ch_3resblocks.pth")
# --------------


def train(data_dir=DATA_DIR, batch_size=BATCH_SIZE, epochs=EPOCHS):
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

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

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
        writer.writerow(["timestamp", "epoch", "chunk", "train_loss", "val_loss", "lr"])

    # Initialize the model, loss function, and optimizer
    model = ChessNet(
        vocab_size=VOCAB_SIZE,
        history_length=HISTORY_LENGTH,
        board_channels=BOARD_CHANNELS,
        hidden_channels=NUM_CHANNELS,
        num_blocks=NUM_RESIDUAL_BLOCKS
        ).to(device)

    # check if resuming training from a checkpoint
    if RESUME_TRAINING:
        if RESUME_MODEL_PATH and os.path.exists(RESUME_MODEL_PATH):
            print(f"Resuming training from checkpoint: {RESUME_MODEL_PATH}")
            checkpoint_weights = torch.load(RESUME_MODEL_PATH, map_location=device)
            try:
                model.load_state_dict(checkpoint_weights)
                print("Model weights loaded successfully.")
            except RuntimeError as e:
                raise RuntimeError(f"Error loading checkpoint weights: {e}")
    else:
        if not RESUME_MODEL_PATH or not os.path.exists(RESUME_MODEL_PATH):
            print(f"Resume model path specified but file not found: {RESUME_MODEL_PATH}")

    value_criterion = nn.MSELoss()
    policy_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)

    # implement learning rate scheduler to reduce lr if validation loss plateaus
    # uses the OneCycleLR scheduler which performs cosine scheduling with a warmup period
    # https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    # first calculate total training steps

    num_files = len(glob.glob(os.path.join(data_dir, "*.parquet")))
    print(f"Found {num_files} parquet files in {data_dir}")

    # Approx 2M rows per file * number of files
    total_rows = num_files * 2_000_000
    total_steps = epochs * (total_rows // batch_size)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
        )

    # set up AMP if running on gpu
    scaler = GradScaler(enabled=use_amp)

    start_time = time.time()
    print("\n Starting Training With hyperparameters:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Learning Rate: {MAX_LR}")
    print(f"  Epochs: {epochs}")
    print(f"  Validation Split: {VALIDATION_SPLIT}")
    print(f"  Number of Channels: {NUM_CHANNELS}")
    print(f"  Number of Residual Blocks: {NUM_RESIDUAL_BLOCKS}")
    print(f"  Training on {num_files} files")
    print(f"  Number of Training Steps: {total_steps}")
    print(f"  Using AMP: {use_amp}")
    print(f"  Start Time: {time.ctime(start_time)}")

    for epoch in range(epochs):
        print(f"\n --- Epoch {epoch + 1}/{epochs} Lr: {optimizer.param_groups[0]['lr']}---")

        # get a loader for each chunk of data
        train_loader, test_loader, n_train, n_test = get_train_test_loaders(
            data_dir,
            batch_size=batch_size,
            validation_split=VALIDATION_SPLIT,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            history_length=HISTORY_LENGTH
        )

        print(f"  Training on {n_train} files")
        # put the model in training mode
        model.train()
        epoch_train_loss = 0.0
        train_chunks = 0

        for i, loader in enumerate(train_loader):
            chunk_loss = 0.0
            steps = 0
            chunk_start_time = time.time()

            for inputs, value_targets, policy_targets in loader:
                # move tensors to the configured device
                inputs = inputs.to(device)
                value_targets = value_targets.to(device)
                policy_targets = policy_targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # check if using AMP, then use autocast to speed up training by using mixed precision where possible
                with autocast(enabled=use_amp, device_type=amp_device, dtype=torch.float16):
                    # Forward pass
                    policy_pred, value_pred = model(inputs)
                    # compute combined loss
                    value_loss = value_criterion(value_pred, value_targets)
                    policy_loss = policy_criterion(policy_pred, policy_targets)
                    loss = value_loss + policy_loss

                # Backward pass and optimization with AMP scaler
                scaler.scale(loss).backward()

                # gradient clipping, prevents exploding gradients by capping max gradient norm
                # unscale before clipping
                scaler.unscale_(optimizer)
                # clip the gradients to max norm of 1.0
                clip_grad_norm(model.parameters(), max_norm=1.0)

                # then step the scalar and the scheduler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                chunk_loss += loss.item()
                steps += 1

            avg_loss = chunk_loss / steps if steps > 0 else 0
            epoch_train_loss += avg_loss
            train_chunks += 1
            current_lr = scheduler.get_last_lr()[0]
            print(f'    [Chunk {i + 1}] Training Loss: {avg_loss:.6f}, LR: {current_lr}, in: {time.time() - chunk_start_time:.2f} seconds')
            with open(log_path, "a") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([f"{time.time() - start_time:.2f}", epoch + 1, i + 1, f"{avg_loss:.6f}", "", f"{current_lr}"])

            # save a model checkpoint every 25 chunks
            if (i + 1) % 25 == 0:
                checkpoint_path = os.path.join(session_model_dir, "latest_checkpoint.pth")
                torch.save({
                    'epoch': epoch,
                    'chunk': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                print(f"    --> Safety checkpoint saved to {checkpoint_path}")

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

                for inputs, value_targets, policy_targets in loader:
                    # move tensors to the configured device
                    inputs = inputs.to(device)
                    value_targets = value_targets.to(device)
                    policy_targets = policy_targets.to(device)

                    # Forward pass
                    policy_pred, value_pred = model(inputs)
                    value_loss = value_criterion(value_pred, value_targets)
                    policy_loss = policy_criterion(policy_pred, policy_targets)
                    loss = value_loss + policy_loss

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
    parser = argparse.ArgumentParser(description="Train the Chess Value Network.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Directory containing processed training data.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size.")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
        )
