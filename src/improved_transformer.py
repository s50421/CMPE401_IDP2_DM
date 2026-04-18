import os
# [NOTE]: We are forcing Keras to use the PyTorch backend here. 
# Why? Because getting Keras 3 to play nicely with native Windows CUDA is a huge pain,
# but PyTorch handles the RTX 4090 perfectly out of the box.
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

# Enable mixed precision for RTX 4090 to train much faster and use less VRAM
keras.mixed_precision.set_global_policy("mixed_float16")

def readucr(filename):
    """
    Helper function to load the UCR TSV format.
    The first column is the label, and the rest of the row is the time-series signal.
    """
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Standard Transformer block for sequence processing.
    """
    # [BUG FIX]: Originally this was Post-LN (LayerNorm after the residual add),
    # which caused massive gradient vanishing and loss stuck at 0.693. 
    # Swapping to Pre-LN (LayerNorm before attention) fixed the convergence issues completely!
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs  # Residual connection

    # Feed Forward Part (Also Pre-LN)
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Stack the transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # [CRITICAL BUG FIX]: The original Keras tutorial used data_format="channels_last".
    # But because the FordA dataset is zero-mean normalized across the 500 timesteps, 
    # pooling across time literally averaged everything to zero! 
    # Using "channels_first" pools across the feature dimension instead, preserving the sequence data.
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    # Since we use mixed_float16, explicitly cast the output to float32 for numerical stability
    outputs = layers.Dense(len(np.unique(y_train)), activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs)

def main(opt):
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    print(f"Loading data from {root_url}...")
    global x_train, y_train
    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    input_shape = x_train.shape[1:]

    print(f"Building Improved Transformer Model...")
    model = build_model(
        input_shape,
        head_size=128,  # Modification 1: Changed head_size from 256 to 128
        num_heads=8,    # Modification 1: Changed num_heads from 4 to 8
        ff_dim=4,
        num_transformer_blocks=6, # Modification 2: Increased blocks from 4 to 6
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.4, # Modification 3: Increased dropout from 0.25 to 0.4
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=opt.lr),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    print(f"Starting training for {opt.epochs} epochs with batch size {opt.batch_size}...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        callbacks=callbacks,
    )

    print("Evaluating on test set...")
    model.evaluate(x_test, y_test, verbose=1)

    print("Generating and saving graphics...")
    plt.figure()
    plt.plot(history.history["sparse_categorical_accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_sparse_categorical_accuracy"], label="Validation Accuracy")
    plt.title("Improved Transformer Accuracy")
    plt.legend()
    plt.savefig("improved_transformer_accuracy.png")

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Improved Transformer Loss")
    plt.legend()
    plt.savefig("improved_transformer_loss.png")
    print("Graphics saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Transformer Training")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    opt = parser.parse_args()
    main(opt)