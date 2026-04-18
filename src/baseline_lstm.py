import os
# [NOTE]: We are forcing Keras to use the PyTorch backend here. 
# Why? Because getting Keras 3 to play nicely with native Windows CUDA is a huge pain,
# but PyTorch handles the RTX 4090 perfectly out of the box.
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import keras
from zipfile import ZipFile

# Enable mixed precision for RTX 4090 to train much faster and use less VRAM
# This uses float16 for calculations but keeps weights in float32
keras.mixed_precision.set_global_policy("mixed_float16")

titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

colors = [
    "blue", "orange", "green", "red", "purple", 
    "brown", "pink", "gray", "olive", "cyan",
]

date_time_key = "Date Time"

def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()

def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    future = delta if delta else 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()

def main(opt):
    uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
    zip_file = ZipFile(zip_path)
    zip_file.extractall()
    csv_path = "jena_climate_2009_2016.csv"

    df = pd.read_csv(csv_path)
    
    # Optional: show_raw_visualization(df)
    
    split_fraction = 0.715
    train_split = int(split_fraction * int(df.shape[0]))
    step = 6
    past = 720
    future = 72

    print("The selected parameters are:", ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]))
    selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
    features = df[selected_features]
    features.index = df[date_time_key]

    features = normalize(features.values, train_split)
    features = pd.DataFrame(features)

    train_data = features.loc[0 : train_split - 1]
    val_data = features.loc[train_split:]
    start = past + future
    end = start + train_split

    x_train = train_data[[i for i in range(7)]].values
    y_train = features.iloc[start:end][[1]]

    sequence_length = int(past / step)
    dataset_train = keras.utils.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=opt.batch_size,
    )
    x_end = len(val_data) - past - future
    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
    y_val = features.iloc[label_start:][[1]]

    dataset_val = keras.utils.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=opt.batch_size,
    )

    import itertools
    for batch in itertools.islice(dataset_train, 1):
        inputs, targets = batch

    # Convert to numpy arrays safely regardless of backend (tf tensor vs torch tensor)
    inputs_np = keras.ops.convert_to_numpy(inputs)
    targets_np = keras.ops.convert_to_numpy(targets)
    
    print("Input shape:", inputs_np.shape)
    print("Target shape:", targets_np.shape)
    
    inputs_layer = keras.layers.Input(shape=(inputs_np.shape[1], inputs_np.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs_layer)
    outputs = keras.layers.Dense(1, dtype="float32")(lstm_out)

    model = keras.Model(inputs=inputs_layer, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=opt.lr), loss="mse")
    model.summary()
    
    path_checkpoint = "model_checkpoint.weights.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        dataset_train,
        epochs=opt.epochs,
        validation_data=dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )

    visualize_loss(history, "Training and Validation Loss")

    for batch in itertools.islice(dataset_val, 5):
        x, y = batch
        x_np = keras.ops.convert_to_numpy(x)
        y_np = keras.ops.convert_to_numpy(y)
        pred = keras.ops.convert_to_numpy(model.predict(x))
        show_plot(
            [x_np[0][:, 1], y_np[0], pred[0]],
            12,
            "Single Step Prediction",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline LSTM Training")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    opt = parser.parse_args()
    main(opt)