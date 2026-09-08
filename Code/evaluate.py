import numpy as np
import tensorflow as tf
import argparse
import json
from pathlib import Path

from data_loader import DataLoader

# ---- GPU availability report ---------------------------------------------- #
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("[GPU] TensorFlow detected " + str(len(gpus)) + " GPU(s):")
    for g in gpus:
        print("      " + str(g))
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            print("[GPU] set_memory_growth failed: " + str(e))
else:
    print("[GPU] No GPU detected -- running inference on CPU.")

# ----------------- Define cml arguments ---------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, default="hr",
                    help="Evaluation mode (hr or spo2)")
parser.add_argument("--dataset", type=str, required=False, default="bidmc",
                    help="Dataset to evaluate on: bidmc or mths")
parser.add_argument("--downsample", type=int, required=False, default=2,
                    help="Down-sample factor (MTHS only)")
parser.add_argument("--timelen", type=int, required=False, default=10,
                    help="Time length of each input signal (MTHS only)")
parser.add_argument("--batchsize", type=int, required=False, default=80,
                    help="Batch size for inference")
parser.add_argument("--testsize", type=float, required=False, default=0.2,
                    help="Test set proportion (MTHS only)")
parser.add_argument("--valsize", type=float, required=False, default=0.15,
                    help="Validation set proportion (MTHS only)")
parser.add_argument("--savedir", type=str, required=False, default="./",
                    help="Directory where trained model checkpoints are stored")

args        = parser.parse_args()
TRAIN_MODE  = args.mode
DATASET_NAME = args.dataset
DOWNSAMPLE_FACTOR = args.downsample
TIME_LENGTH = args.timelen
TEST_SIZE   = args.testsize
VALID_SIZE  = args.valsize
saving_dir  = args.savedir
BATCH_SIZE  = args.batchsize

MODEL_NAMES = ("BASE", "FCN", "FCN_Residual", "FCN_DCT")


def compute_metrics(y_true, y_pred):
    y_true = y_true.ravel().astype(np.float64)
    y_pred = y_pred.ravel().astype(np.float64)
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    mse  = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    pearson_r = (float(np.corrcoef(y_true, y_pred)[0, 1])
                 if y_true.std() > 0 and y_pred.std() > 0 else float("nan"))
    mean_error = float(np.mean(y_pred - y_true))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse,
            "R2": r2, "Pearson_r": pearson_r, "Mean_Error": mean_error}


def print_metrics(model_name, metrics):
    print(
        "[" + model_name + "] "
        "MAE: {MAE:.4f} | MSE: {MSE:.4f} | RMSE: {RMSE:.4f} | "
        "R2: {R2:.4f} | Pearson r: {Pearson_r:.4f} | "
        "Bias (Mean Error): {Mean_Error:.4f}".format(**metrics)
    )


if __name__ == "__main__":
    assert TRAIN_MODE in ["hr", "spo2"], "Training mode must be 'hr' or 'spo2'."

    loader = DataLoader(mode=TRAIN_MODE)

    if DATASET_NAME == "bidmc":
        x_train, y_train, x_valid, y_valid, x_test, y_test = loader.load_bidmc()
    elif DATASET_NAME == "mths":
        x_train, y_train, x_valid, y_valid, x_test, y_test = loader.load_mths(
            dwn_factor=DOWNSAMPLE_FACTOR,
            time_length=TIME_LENGTH,
            testset_size=TEST_SIZE,
            validset_proportion=VALID_SIZE,
        )
    else:
        print("Wrong dataset name - Please check and try again")
        exit()

    loss_name  = "mean_squared_error"
    root = Path(saving_dir)
    results_dir = root / "results" / DATASET_NAME / TRAIN_MODE
    results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for model_name in MODEL_NAMES:
        checkpoint_path = root / "checkpoints" / DATASET_NAME / TRAIN_MODE / model_name / "best.h5"
        print("\nEvaluating " + model_name + " (loss=" + loss_name + ")")
        print("-" * 65)
        if not checkpoint_path.exists():
            print("[ERROR] Missing checkpoint: " + str(checkpoint_path))
            continue
        try:
            model = tf.keras.models.load_model(str(checkpoint_path))
            y_pred = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).ravel()
            metrics = compute_metrics(y_test, y_pred)
        except Exception as e:
            print("[ERROR] Could not evaluate " + model_name + ": " + str(e))
            continue
        all_metrics[model_name] = metrics
        print_metrics(model_name, metrics)
        print("-" * 65)

    results_path = results_dir / "metrics.json"
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(all_metrics, results_file, indent=2, allow_nan=True)
    print("\n[evaluate] Metrics saved to: " + str(results_path))
    if len(all_metrics) != len(MODEL_NAMES):
        raise SystemExit("[evaluate] Evaluation incomplete: " + str(len(all_metrics)) + "/" + str(len(MODEL_NAMES)) + " models")
