import json, os, random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

_MEDVSE_ROOT = Path(__file__).resolve().parents[1]
_BIDMC_CSV_PATH = str(
    _MEDVSE_ROOT / "datasets/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv"
)
_BIDMC_SPLIT_INFO_PATH = str(_MEDVSE_ROOT / "configs/bidmc_split_info.json")


def _bandpass_filter(data, lowcut=0.5, highcut=5.0, fs=125, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def _resample_to_input_dim(values, input_dim):
    """Match STAG-HR by linearly resampling each native PPG window."""
    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("Cannot resample an empty PPG window")
    if signal.size == input_dim:
        return signal
    if signal.size == 1:
        return np.full(input_dim, signal.item(), dtype=np.float32)
    source_grid = np.linspace(0.0, 1.0, num=signal.size, dtype=np.float32)
    target_grid = np.linspace(0.0, 1.0, num=input_dim, dtype=np.float32)
    return np.interp(target_grid, source_grid, signal).astype(np.float32)


def _split_subjects(subject_ids, seed=36, val_split=0.2, test_split=0.2):
    n = len(subject_ids)
    rng = random.Random(seed)
    shuffled = subject_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(n * val_split))
    n_test = max(1, round(n * test_split))
    while n_val + n_test >= n:
        if n_val >= n_test:
            n_val -= 1
        else:
            n_test -= 1
    test_subjects = set(shuffled[:n_test])
    val_subjects = set(shuffled[n_test:n_test + n_val])
    train_subjects = set(shuffled[n_test + n_val:])
    return train_subjects, val_subjects, test_subjects

def _split_subjects_multi_seed(subject_ids, seeds=[36, 42, 99, 300, 900, 1400], val_split=0.2, test_split=0.2):
    split_results = []
    for seed in seeds:
        train, val, test = _split_subjects(subject_ids, seed=seed, val_split=val_split, test_split=test_split)
        split_results.append((train, val, test))
    return split_results

def _load_split_info(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if not {"train_subjects", "val_subjects", "test_subjects"}.issubset(info):
            return None
        return info
    except Exception as e:
        print("[split] Warning: could not load split_info.json (" + str(e) + ")")
        return None


class DataLoader():
    def __init__(self, root_path="../", mode="hr"):
        self.root_path = root_path
        self.mode = mode

    def load_mths(self, dwn_factor=2, time_length=10, testset_size=0.2, validset_proportion=0.15):
        mred, mblue, mgreen = [], [], []
        hr, spo2 = [], []
        seq_len = int(30 / dwn_factor) * time_length
        print("Each sample input signal sequence length would be " + str(seq_len))
        for i in range(67):
            try:
                signals = np.load(self.root_path + "/MTHS/Data/signal_{}.npy".format(i))
                labels  = np.load(self.root_path + "/MTHS/Data//label_{}.npy".format(i))
            except:
                continue
            mred.extend(signals[:, 0])
            mgreen.extend(signals[:, 1])
            mblue.extend(signals[:, 2])
            hr.extend(labels[:, 0])
            spo2.extend(labels[:, 1])
        mred, mblue, mgreen = mred[::dwn_factor], mblue[::dwn_factor], mgreen[::dwn_factor]
        hr_t   = np.zeros((len(mred) // seq_len,))
        spo2_t = np.zeros((len(mred) // seq_len,))
        if self.mode == "hr":
            ppg_t = np.zeros((len(mred) // seq_len, seq_len, 1))
        else:
            ppg_t = np.zeros((len(mred) // seq_len, seq_len, 3))
        for i in range(ppg_t.shape[0]):
            hr_t[i]   = np.mean(hr[i * time_length:(i + 1) * time_length])
            spo2_t[i] = np.mean(spo2[i * time_length:(i + 1) * time_length])
            s, e = i * seq_len, (i + 1) * seq_len
            if self.mode == "hr":
                ppg_t[i, :, 0] = mred[s:e]
            else:
                ppg_t[i, :, 0] = mred[s:e]
                ppg_t[i, :, 1] = mblue[s:e]
                ppg_t[i, :, 2] = mgreen[s:e]
        labels_t = hr_t if self.mode == "hr" else spo2_t
        x_train, x_test, y_train, y_test = train_test_split(
            ppg_t, labels_t, test_size=testset_size, random_state=1400)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=validset_proportion, random_state=1400)
        print("X, y train shapes = ", (x_train.shape, y_train.shape))
        print("X, y valid shapes = ", (x_valid.shape, y_valid.shape))
        print("X, y test shapes = ",  (x_test.shape,  y_test.shape))
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def load_bidmc(self, data_path=_BIDMC_CSV_PATH, window_size=16, window_shift=2,
                   sampling_rate=125, eps=1e-8, seed=36, val_split=0.2, test_split=0.2,
                   input_dim=1000, split_info_path=_BIDMC_SPLIT_INFO_PATH, **_ignored):
        data_path = Path(data_path)
        print("[BIDMC] Loading CSVs from " + str(data_path) + " ...")
        subject_records = {}
        for idx in range(1, 54):
            sid  = "bidmc_{:02d}".format(idx)
            sp   = data_path / (sid + "_Signals.csv")
            np_  = data_path / (sid + "_Numerics.csv")
            if not sp.exists() or not np_.exists():
                print("[BIDMC] Warning: missing CSVs for " + sid + ", skipping")
                continue
            try:
                sig_df = pd.read_csv(sp)
                num_df = pd.read_csv(np_)
                sig_df.columns = sig_df.columns.str.strip()
                num_df.columns = num_df.columns.str.strip()
            except Exception as exc:
                print("[BIDMC] Warning: failed to read " + sid + " (" + str(exc) + "), skipping")
                continue
            if "PLETH" not in sig_df.columns or "HR" not in num_df.columns or "Time [s]" not in num_df.columns:
                print("[BIDMC] Warning: missing required columns for " + sid + ", skipping")
                continue
            subject_records[sid] = {
                "ppg":  sig_df["PLETH"].values,
                "hr":   num_df["HR"].values,
                "time": num_df["Time [s]"].values,
            }
        if not subject_records:
            raise RuntimeError("[BIDMC] No subjects were successfully loaded.")
        print("[BIDMC] Loaded " + str(len(subject_records)) + " subjects.")
        all_windows = {}
        for sid, rec in subject_records.items():
            ppg_signal  = rec["ppg"]
            hr_values   = rec["hr"]
            time_values = rec["time"]
            total_seconds = len(ppg_signal) // sampling_rate
            windows = []
            for start_sec in range(0, total_seconds - window_size + 1, window_shift):
                end_sec  = start_sec + window_size
                win_ppg  = ppg_signal[start_sec * sampling_rate: end_sec * sampling_rate]
                win_hr   = hr_values[(time_values >= start_sec) & (time_values < end_sec)]
                hr_label = float(np.nanmean(win_hr)) if len(win_hr) > 0 else None
                if hr_label is None or np.isnan(hr_label):
                    continue
                win_ppg = _bandpass_filter(win_ppg, fs=sampling_rate)
                win_ppg = (win_ppg - np.mean(win_ppg)) / (np.std(win_ppg) + eps)
                windows.append((win_ppg.astype(np.float32), np.float32(hr_label)))
            if windows:
                all_windows[sid] = windows
        subject_ids = sorted(all_windows.keys())
        split_info = _load_split_info(split_info_path)
        if split_info is not None:
            train_subjects = set(split_info["train_subjects"])
            val_subjects   = set(split_info["val_subjects"])
            test_subjects  = set(split_info["test_subjects"])
            missing = (train_subjects | val_subjects | test_subjects) - set(subject_ids)
            if missing:
                print("[split] Warning: subjects in split_info not found in data: " + str(sorted(missing)))
            print("[split] Reused BL-PPG split_info.json -> "
                  + str(len(train_subjects)) + " train / "
                  + str(len(val_subjects))   + " val / "
                  + str(len(test_subjects))  + " test subjects")
        else:
            train_subjects, val_subjects, test_subjects = _split_subjects(
                subject_ids, seed=seed, val_split=val_split, test_split=test_split)
            print("[split] Generated new seeded split -> "
                  + str(len(train_subjects)) + " train / "
                  + str(len(val_subjects))   + " val / "
                  + str(len(test_subjects))  + " test subjects")

        def _collect(sids):
            xs, ys = [], []
            for s in subject_ids:
                if s in sids and s in all_windows:
                    for ppg_w, hr_l in all_windows[s]:
                        xs.append(_resample_to_input_dim(ppg_w, input_dim))
                        ys.append(hr_l)
            return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

        x_train, y_train = _collect(train_subjects)
        x_valid, y_valid = _collect(val_subjects)
        x_test,  y_test  = _collect(test_subjects)
        x_train = x_train.reshape(-1, input_dim, 1)
        x_valid = x_valid.reshape(-1, input_dim, 1)
        x_test  = x_test.reshape(-1, input_dim, 1)
        print("X, y train shapes = ", (x_train.shape, y_train.shape))
        print("X, y valid shapes = ", (x_valid.shape, y_valid.shape))
        print("X, y test shapes = ",  (x_test.shape,  y_test.shape))
        return x_train, y_train, x_valid, y_valid, x_test, y_test
