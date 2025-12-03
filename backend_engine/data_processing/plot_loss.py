import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(log_file='../ml/models/session_1764625687/training_log.csv'):
    df = pd.read_csv(log_file)

    # Determine max chunk size (ignoring 'end')
    numeric_chunks = df[df['chunk'] != 'end']['chunk'].astype(int)
    max_chunk = numeric_chunks.max()

    # Create a continuous x-axis (global step)
    def get_global_step(row):
        if row['chunk'] == 'end':
            return row['epoch'] * max_chunk
        else:
            return (row['epoch'] - 1) * max_chunk + int(row['chunk'])

    df['global_step'] = df.apply(get_global_step, axis=1)

    # Separate training and validation losses
    train_df = df[df['train_loss'].notna()]
    val_df = df[df['val_loss'].notna()]

    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(train_df['global_step'], train_df['train_loss'], label='Training Loss', marker='o', linestyle='-')

    # Plot validation loss
    plt.plot(val_df['global_step'], val_df['val_loss'], label='Validation Loss', marker='X', markersize=10, linestyle='None', color='orange')

    plt.xlabel('Global Step (Cumulative Chunks)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_loss()
