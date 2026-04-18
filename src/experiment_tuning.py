import os
# [NOTE]: We are forcing Keras to use the PyTorch backend here. 
# Why? Because getting Keras 3 to play nicely with native Windows CUDA is a huge pain,
# but PyTorch handles the RTX 4090 perfectly out of the box.
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import numpy as np
import pandas as pd
import keras
from keras import layers
import matplotlib.pyplot as plt
import torch

# Enable mixed precision for RTX 4090 to train much faster and use less VRAM.
# This uses float16 for calculations but keeps weights in float32 so we don't lose precision.
keras.mixed_precision.set_global_policy("mixed_float16")

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention (Pre-LN)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part (Pre-LN)
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
    num_classes=2
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Use channels_first so we pool across the feature dimension and preserve the sequence
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    # Cast to float32 for numerical stability with mixed precision
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs)

def main(opt):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    print(f"Loading data from {root_url}...")
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
    num_classes = len(np.unique(y_train))

    # Pick 3 random samples for visualization
    sample_indices = np.random.choice(len(x_test), 3, replace=False)
    sample_x = x_test[sample_indices]
    sample_y = y_test[sample_indices]
    sample_predictions = {}

    # Define hyperparameter configurations for tuning
    configurations = {
        "Baseline": {"blocks": 4, "heads": 4, "head_size": 256, "dropout": 0.25},
        "Wider_Attention": {"blocks": 4, "heads": 8, "head_size": 128, "dropout": 0.3},
        "Deeper_Network": {"blocks": 6, "heads": 4, "head_size": 256, "dropout": 0.3},
        "Deep_and_Wide": {"blocks": 6, "heads": 8, "head_size": 128, "dropout": 0.4},
        "Lightweight": {"blocks": 2, "heads": 4, "head_size": 128, "dropout": 0.2},
    }

    results = []

    for name, config in configurations.items():
        print(f"\n{'='*50}")
        print(f"Training Configuration: {name}")
        print(f"{config}")
        print(f"{'='*50}")

        model = build_model(
            input_shape,
            head_size=config["head_size"],
            num_heads=config["heads"],
            ff_dim=4,
            num_transformer_blocks=config["blocks"],
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=config["dropout"],
            num_classes=num_classes
        )

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=opt.lr),
            metrics=["sparse_categorical_accuracy"],
        )

        callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

        history = model.fit(
            x_train,
            y_train,
            validation_split=0.2,
            epochs=opt.epochs,
            batch_size=opt.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print(f"\nEvaluating {name} on test set...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        results.append({
            "Configuration": name,
            "Blocks": config["blocks"],
            "Heads": config["heads"],
            "Head_Size": config["head_size"],
            "Dropout": config["dropout"],
            "Test_Loss": test_loss,
            "Test_Accuracy": test_acc
        })

        print(f"Generating predictions for {name} on sample data...")
        preds = model.predict(sample_x)
        sample_predictions[name] = np.argmax(preds, axis=1)

        # Save individual convergence graphics
        print(f"Generating and saving graphics for {name}...")
        plt.figure()
        plt.plot(history.history["sparse_categorical_accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_sparse_categorical_accuracy"], label="Validation Accuracy")
        plt.title(f"{name.replace('_', ' ')} Accuracy")
        plt.legend()
        plt.savefig(f"experiment_{name}_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"{name.replace('_', ' ')} Loss")
        plt.legend()
        plt.savefig(f"experiment_{name}_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n{'='*50}")
    print("All configurations complete! Generating summary...")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("experiment_results.csv", index=False)
    print("Results saved to experiment_results.csv")

    # Generate comparative bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df["Configuration"].str.replace('_', '\n'), results_df["Test_Accuracy"], color='skyblue')
    plt.title("Comparative Benchmark: Test Accuracy by Configuration")
    plt.ylabel("Test Accuracy")
    plt.ylim([0.0, 1.0])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.savefig("experiment_comparison.png", dpi=300, bbox_inches="tight")
    print("Comparison bar chart saved to experiment_comparison.png")

    # Generate prediction visualization
    print("Generating prediction visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Model Predictions vs Ground Truth on Sample Time-Series", fontsize=16, fontweight='bold')
    
    for i in range(3):
        ax = axes[i]
        ax.plot(sample_x[i].flatten(), color='royalblue', alpha=0.8, linewidth=1.5)
        
        # Build text string for predictions
        truth = sample_y[i]
        pred_str = f"Ground Truth: Class {truth}\n{'-'*25}\n"
        for name in configurations.keys():
            p = sample_predictions[name][i]
            match = "✅" if p == truth else "❌"
            pred_str += f"{name.replace('_', ' ')}: Class {p} {match}\n"
            
        ax.set_title(f"Test Sample {i+1}", fontsize=12)
        ax.text(1.02, 0.5, pred_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='gray', alpha=0.8))
        ax.grid(axis='both', linestyle='--', alpha=0.4)
        ax.set_xlim([0, len(sample_x[i])])
        
    plt.tight_layout(rect=[0, 0, 0.80, 0.96]) # Leave room for the text box on the right
    plt.savefig("experiment_predictions.png", dpi=300, bbox_inches="tight")
    print("Prediction visualization saved to experiment_predictions.png")

    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Experimental Tuning")
    parser.add_argument("--epochs", type=int, default=25, help="number of training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    opt = parser.parse_args()
    main(opt)
