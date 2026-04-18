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
        
    outputs = layers.Dense(len(np.unique(y_train)), activation="softmax")(x)
    return keras.Model(inputs, outputs)

def main(opt):
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    print(f"Loading data from {root_url}...")
    global x_train, y_train  # Needed because build_model expects y_train for np.unique(y_train)
    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    # Reshape the data for Conv1D and Attention layers (samples, timesteps, features)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Shuffle the training split to prevent the model from memorizing sequence order
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    # The dataset uses -1 and 1 for classes. Keras sparse categorical needs 0 and 1.
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    input_shape = x_train.shape[1:]

    print(f"Building Transformer Model...")
    # These are the baseline hyperparameters. We will tweak these later in the improved model.
    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=opt.lr),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    # Automatically stop training if the validation loss doesn't improve for 10 epochs
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

    # Save out the plots for the README report
    print("Generating and saving graphics...")
    plt.figure()
    plt.plot(history.history["sparse_categorical_accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_sparse_categorical_accuracy"], label="Validation Accuracy")
    plt.title("Baseline Transformer Accuracy")
    plt.legend()
    plt.savefig("baseline_accuracy.png")

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Baseline Transformer Loss")
    plt.legend()
    plt.savefig("baseline_loss.png")
    print("Graphics saved successfully.")

if __name__ == "__main__":
    # Setup argparse so we can easily test different epoch/batch sizes from the command line
    parser = argparse.ArgumentParser(description="Baseline Transformer Training")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    opt = parser.parse_args()
    main(opt)