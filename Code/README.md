# Codes for MEDVSE
Official tf-keras implementation of "Efficient Deep Learning-based Estimation of the Vital Signs on Smartphones".

# Usage
This version is configured for the BIDMC dataset only. Run the commands from
the MEDVSE root directory so that the dataset and checkpoint paths are resolved
correctly.

## 1. Install dependencies
```bash
python3 -m pip install -r Code/requirements.txt
```

## 2. Train on BIDMC
```bash
cd /home/pham-anh-khoi/Study/Others/MEDVSE

python Code/train.py \
  --mode hr \
  --dataset bidmc \
  --batchsize 128 \
  --epochs 150 \
  --seed 36 \
  --lr 0.003 \
  --weight-decay 0.0001 \
  --lr-factor 0.5 \
  --lr-patience 8 \
  --savedir ./
```

The four best checkpoints are saved in a deterministic layout:
`./checkpoints/bidmc/hr/<MODEL>/best.h5`, where `<MODEL>` is `BASE`, `FCN`,
`FCN_Residual`, or `FCN_DCT`. Training histories are saved under
`./histories/bidmc/hr/<MODEL>/history.json`.

For BIDMC, the loader uses 16-second windows with a 2-second shift. Each
native 2000-sample window at 125 Hz is resampled to 1000 samples before it is
passed to the model. Train, validation, and test subjects use the fixed split
in `configs/bidmc_split_info.json`.

## 3. Evaluate on the BIDMC test set
Run this after training:
```bash
python Code/evaluate.py \
  --mode hr \
  --dataset bidmc \
  --batchsize 128 \
  --savedir ./
```

The evaluation command loads all four checkpoints and reports MAE, MSE, RMSE,
R2, Pearson correlation, and mean error on the fixed BIDMC test subjects.
The combined result is saved as `./results/bidmc/hr/metrics.json`.

### Available arguments for BIDMC
* `--mode`: `hr` is the supported mode for this comparison.
* `--dataset`: Use `bidmc`.
* `--batchsize`: Batch size for training or evaluation.
* `--epochs`: Number of training epochs; used only by `train.py`.
* `--seed`: Random seed for reproducible training; default is `36`.
* `--savedir`: Directory containing or receiving the checkpoint.

For a stable comparison with STAG-HR, keep the same seed, batch size, epochs,
learning rate, AdamW weight decay, learning-rate scheduler, MSE loss, 1000-sample
input, and fixed subject split. The scheduler monitors validation loss and
reduces the learning rate by `--lr-factor` after `--lr-patience` stagnant epochs.
The evaluation batch size only affects speed and memory for an existing checkpoint.

The arguments `--downsample`, `--timelen`, `--testsize`, and `--valsize` are
only used by the legacy MTHS path and should not be passed for BIDMC.


# Citation
Please cite our work in your project:
```
@article{samavati2022efficient,
  title={Efficient deep learning-based estimation of the vital signs on smartphones},
  author={Samavati, Taha and Farvardin, Mahdi and Ghaffari, Aboozar},
  journal={arXiv preprint arXiv:2204.08989},
  year={2022}
}

```
[Efficient Deep Learning-based Estimation of the Vital Signs on Smartphones](https://arxiv.org/abs/2204.08989)
