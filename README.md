<!-- # balle2018-scale-hyperprior -->

# Variational Image Compression with a Scale Hyperprior (Ballé et al., 2018)

This repository contains an unofficial TensorFlow 2.x implementation of the paper [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/pdf/1802.01436) by Ballé et al. (2018), a foundational work in the field of learned image compression.

## Features

- Implements the scale hyperprior architecture for variational image compression.
- Supports training with large-scale datasets (e.g., ImageNet).
- Modular training utilities with callbacks for checkpointing, early stopping, and image logging.
- Configurable via command-line arguments for flexible experimentation.

## Repository Structure

```
balle2018-scale-hyperpior
  ├── README.md
  ├── LICENSE
  ├── data/
      └── imagenet_patch_loader.py
  ├── models/
      └── balle_hyperprior_2018_model.py
  └── training/
      ├── train.py
      └── callbacks.py
      └── losses.py
```

---

## Usage

### Training

The main entry point is `training/train.py`. 

#### Command-line Arguments

Key arguments:
- `--data_dir`: Directory or .tar file containing images (required).
- `--patch_size`: Size of image patches for training and validation (default: 256).
- `--batch_size`: Batch size for training (default: 16).
- `--patches_per_image`: Number of patches to extract per image (default: 1).
- `--max_images`: Maximum number of images to use (default: None; uses all images found).
- `--val_split`: Fraction of images to use for validation (default: 0.1).
- `--test_split`: Fraction of images to use for testing (default: 0.1).

Model Arguments:
- `--main_channels`: Number of channels in the main encoder/decoder (default: 192).
- `--hyper_channels`: Number of channels in the hyper encoder/decoder (default: 128).

Training Arguments:
- `--epochs`: Number of training epochs (default: 100).
- `--num_scales`: Number of Gaussian scales for range coding tables (default: 64).
- `--scale_min`: Minimum standard deviation for Gaussian scales (default: 0.11).
- `--scale_max`: Maximum standard deviation for Gaussian scales (default: 256).
- `--learning_rate`: Learning rate for Adam optimizer (default: 1e-4).
- `--lambda_val`: Trade-off parameter between rate and distortion (default: 0.01).
- `--metric`: Distortion metric to optimize, choices are "mse" or "ms-ssim" (default: "mse").
- `--save_dir`: Directory to save models and logs (default: "saved_model").

Callbacks Arguments:
- `--save_every_epochs`: Save a model checkpoint every N epochs (default: 5).
- `--early_stopping_patience`: Stop training if validation loss does not improve for N epochs (default: 5).
- `--log_images_every`: Log original vs reconstructed images every N epochs (default: 0; disables logging).
- `--num_images_to_log`: Number of images to log when logging is enabled (default: 4).

Example usage:
```bash

python training/train.py \
  --data_dir /path/to/data \
  --patch_size 256 \
  --batch_size 16 \
  --patches_per_image 2 \
  --main_channels 192 \
  --hyper_channels 128 \
  --epochs 200 \
  --learning_rate 1e-4 \
  --lambda_val 0.01 \
  --metric ms-ssim \
  --save_every_epochs 5 \
  --early_stopping_patience 5 \
  --log_images_every 10 \
  --num_images_to_log 4
```

---

## Core Components

### 1. Data Loading: `data/imagenet_patch_loader.py`

- **Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- **Features**:
  - Extracts datasets from `.tar` archives.
  - Splits dataset into train, validation, and test sets.
  - Builds TensorFlow datasets of image patches (random crops).

**Main Functions:**
- `unzip_tar_file`: Extracts images from tar files.
- `list_images`: Lists images in directory.
- `build_patch_dataset`: Converts image paths into TensorFlow dataset of patches.
- `create_train_val_test_datasets`: Returns train, val, test datasets (randomly split).

### 2. Model: `models/balle_hyperprior_2018_model.py`

- Implements the Ballé et al. (2018) Scale Hyperprior Model for Variational Image Compression.
- Key Class:
- Balle2018Hyperprior(tf.keras.Model): The main model class encapsulating architecture, compression logic, and training routines.
- Architecture Components: Main Encoder (g_a), Hyper Encoder (h_a), Hyper Decoder (h_s), Main Decoder (g_s).
- Entropy Models:
  - Main Entropy Model: LocationScaleIndexedEntropyModel for compressing the main latents conditioned on predicted scales.
  - Hyperprior Entropy Model: ContinuousBatchedEntropyModel for compressing hyper-latents.
<!--
Key Methods:
- call(x, training=False): Forward pass; compresses and reconstructs images; returns reconstructed images, main bits, and hyperprior bits.
- train_step(data): Custom training step for rate-distortion optimization; computes gradients and updates weights.
- test_step(data): Evaluation step; calculates loss and metrics on validation/test data.
- compress(x): Compresses an image and returns the bit strings and shape required for decompression.
- decompress(strings, shape): Decompresses bit strings to reconstruct the image.
-->
- Loss Function: Uses a weighted rate-distortion loss (combining bits-per-pixel and distortion, e.g., MSE or MS-SSIM).
- Easily configurable via constructor arguments for different datasets or compression settings.

### 3. Training Script: `training/train.py`

- Parses arguments.
- Loads and splits data.
- Initializes the model with user-specified parameters.
- Compiles model with Adam optimizer.
- Sets up callbacks (checkpointing, early stopping, image logging).
- Trains the model, saves final weights, evaluates on test set.

### 4. Callbacks: `training/callbacks.py`

- **KeepLastNCheckpoints**: Only keeps the last N checkpoints during training.
- **ImageLoggingCallback**: Logs original vs reconstructed images at specified intervals, saving side-by-side comparisons.
- **EarlyStopping**: Stops training when validation loss stops improving.
- **TerminateOnNaN**: Stops training if NaN loss encountered.

### 5. Loss Function: `training/losses.py`
- Implements the core rate-distortion loss for neural compression.
- **compute_rate_distortion_loss** combines bits-per-pixel (rate) and distortion (MSE or MS-SSIM).
- Balances compression rate and reconstruction quality via a lambda parameter (`lambda_rd`) and selectable metric (`mse` or `ms-ssim`).

---

## Example Workflow

1. **Prepare Dataset**: Place images in a directory or compress into a tar file.
2. **Configure Training**: Adjust arguments for patch size, batch size, model architecture, etc.
3. **Train Model**: Run `train.py` with desired arguments.
4. **Monitor Training**: Checkpoints, early stopping, and image logs help monitor progress.
5. **Evaluate Model**: Final evaluation is performed on the held-out test set.

---

## Outputs

- **Checkpoints**: Saved every N epochs and only the last 5 are retained.
- **Final Model**: Saved in the specified save directory.
- **Image Logs**: Side-by-side original and reconstructed images for visual inspection.
- **Test Results**: Prints loss, bits-per-pixel (BPP), and distortion metrics.

---


## References

- Ballé, J., Minnen, D., Singh, S., & Toderici, G. (2018). "Variational Image Compression with a Scale Hyperprior." [Paper link](https://arxiv.org/pdf/1802.01436)

---

## Contributing

Feel free to fork the repo and submit pull requests.
