import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

SEED = 36
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

from data_loader import DataLoader
from models.models import Model_vault

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
    print("[GPU] No GPU detected -- training on CPU.")

# ----------------- Define cml arguments ---------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, default="hr",
                    help="The training mode (hr or spo2)")
parser.add_argument("--dataset", type=str, required=False, default="bidmc",
                    help="Dataset to train on: bidmc or mths")
parser.add_argument("--downsample", type=int, required=False, default=2,
                    help="Down-sample factor for MTHS PPG signals")
parser.add_argument("--timelen", type=int, required=False, default=10,
                    help="Time length of each input signal (MTHS only)")
parser.add_argument("--batchsize", type=int, required=False, default=80,
                    help="Batch size (default 80 -- matches STAG-HR bidmc.yaml)")
parser.add_argument("--epochs", type=int, required=False, default=150,
                    help="Number of training epochs (default 150 -- matches BL-PPG bidmc.yaml)")
parser.add_argument("--testsize", type=float, required=False, default=0.2,
                    help="Test set proportion (MTHS only)")
parser.add_argument("--valsize", type=float, required=False, default=0.15,
                    help="Validation set proportion of train data (MTHS only)")
parser.add_argument("--savedir", type=str, required=False, default="./",
                    help="Directory to save trained model checkpoints")
parser.add_argument("--seed", type=int, required=False, default=SEED,
                    help="Random seed for reproducible model training")
parser.add_argument("--lr", type=float, required=False, default=3e-3,
                    help="Initial learning rate")
parser.add_argument("--weight-decay", type=float, required=False, default=1e-4,
                    help="AdamW decoupled weight decay")
parser.add_argument("--lr-factor", type=float, required=False, default=0.5,
                    help="Factor used when reducing the learning rate")
parser.add_argument("--lr-patience", type=int, required=False, default=8,
                    help="Epochs without validation improvement before reducing the learning rate")

# ------------- Parse cml arguments and set config ------------------ #
args = parser.parse_args()
TRAIN_MODE      = args.mode
DATASET_NAME    = args.dataset
DOWNSAMPLE_FACTOR = args.downsample
TIME_LENGTH     = args.timelen
TEST_SIZE       = args.testsize
VALID_SIZE      = args.valsize
saving_dir      = args.savedir
BATCH_SIZE      = args.batchsize
EPOCHS          = args.epochs
SEED            = args.seed
LEARNING_RATE   = args.lr
WEIGHT_DECAY    = args.weight_decay
LR_FACTOR       = args.lr_factor
LR_PATIENCE     = args.lr_patience

MODEL_NAMES = ("BASE", "FCN", "FCN_Residual", "FCN_DCT")


if __name__ == "__main__":
    assert TRAIN_MODE in ["hr", "spo2"], "The training mode must be one of 'hr' or 'spo2'."

    # ---- Load dataset -------------------------------------------------------- #
    loader = DataLoader(mode=TRAIN_MODE)
    x_train, y_train, x_valid, y_valid, x_test, y_test = None, None, None, None, None, None

    if DATASET_NAME == "bidmc":
        # Uses the same BIDMC handler logic + subject-level split as the BL-PPG
        # trainer so that both models are trained/evaluated on identical data.
        # window_size=16, window_shift=2 match configs/train_configs/bidmc.yaml.
        x_train, y_train, x_valid, y_valid, x_test, y_test = loader.load_bidmc()

    elif DATASET_NAME == "mths":
        SEQ_LEN = int(30 / DOWNSAMPLE_FACTOR) * TIME_LENGTH
        x_train, y_train, x_valid, y_valid, x_test, y_test = loader.load_mths(
            dwn_factor=DOWNSAMPLE_FACTOR,
            time_length=TIME_LENGTH,
            testset_size=TEST_SIZE,
            validset_proportion=VALID_SIZE,
        )
    else:
        print("Wrong dataset name - Please check and try again")
        exit()

    # Derive sequence length from the loaded data (works for both datasets).
    seq_len = x_train.shape[1]

    loss = "mean_squared_error"

    root = Path(saving_dir)
    model_builders = (
        ("BASE", "create_base_model"),
        ("FCN", "create_fcn"),
        ("FCN_Residual", "create_fcn_residual"),
        ("FCN_DCT", "create_fcn_dct"),
    )
    for model_name, builder_name in model_builders:
        random.seed(SEED)
        np.random.seed(SEED)
        tf.keras.utils.set_random_seed(SEED)
        model_vault = Model_vault(seq_len, mode=TRAIN_MODE)
        model = getattr(model_vault, builder_name)()
        model_dir = root / "checkpoints" / DATASET_NAME / TRAIN_MODE / model_name
        history_dir = root / "histories" / DATASET_NAME / TRAIN_MODE / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "best.h5"

        print("\nTraining " + model_name + " with loss=" + loss)
        print("=" * 65)
        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            mode="min",
            verbose=1,
        )
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            ),
            loss=loss,
            metrics=["mae"],
        )
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
            validation_data=(x_valid, y_valid),
            callbacks=[callback, lr_scheduler],
        )
        serializable_history = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        with (history_dir / "history.json").open("w", encoding="utf-8") as history_file:
            json.dump(serializable_history, history_file, indent=2)
        print("[train] Best checkpoint saved to: " + str(checkpoint_path))
