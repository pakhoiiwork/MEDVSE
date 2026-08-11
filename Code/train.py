import tensorflow as tf
import argparse

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
                    help="Batch size (default 80 -- matches BL-PPG bidmc.yaml)")
parser.add_argument("--epochs", type=int, required=False, default=150,
                    help="Number of training epochs (default 150 -- matches BL-PPG bidmc.yaml)")
parser.add_argument("--testsize", type=float, required=False, default=0.2,
                    help="Test set proportion (MTHS only)")
parser.add_argument("--valsize", type=float, required=False, default=0.15,
                    help="Validation set proportion of train data (MTHS only)")
parser.add_argument("--savedir", type=str, required=False, default="./",
                    help="Directory to save trained model checkpoints")

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

    # ---- Build model vault and select FCN_Residual only -------------------- #
    # NOTE: Only FCN_Residual is trained here. Other models (BASE, FCN, FCN_DCT)
    # are disabled so that MEDVSE and BL-PPG can be compared under identical
    # conditions with the same model family.
    model_vault = Model_vault(seq_len, mode=TRAIN_MODE)
    model = model_vault.create_fcn_residual()
    model_name = "FCN_Residual"

    # ---- Train with MSE loss (closest to BL-PPG primary criterion) --------- #
    loss = "mean_squared_error"

    print("\nTraining " + model_name + " with loss=" + loss)
    print("=" * 65)

    checkpoint_path = (
        saving_dir + "/" + DATASET_NAME + "_" + TRAIN_MODE + "_" + model_name + "_" + loss + ".h5"
    )
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="auto",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["mae"],
    )

    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        validation_data=(x_valid, y_valid),
        callbacks=[callback],
    )

    print("\n[train] Best checkpoint saved to: " + checkpoint_path)
