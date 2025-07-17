import os
import argparse
import tensorflow as tf
from models.balle_hyperprior_2018_model import Balle2018Hyperprior
from training.callbacks import get_callbacks
from data.imagenet_patch_loader import create_train_val_test_datasets

def main(args):
    # load data after splitting into train-val-test
    train_ds, val_ds, test_ds = create_train_val_test_datasets(
        input_path=args.data_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        max_images=args.max_images,
        val_split=args.val_split,
        test_split=args.test_split,
        patches_per_image=args.patches_per_image,
    )

    # Balle2018 model with lambda and metric
    model = Balle2018Hyperprior(
        main_channels = args.main_channels,
        hyper_channels = args.hyper_channels, 
        lambda_rd = args.lambda_val, 
        num_scales = args.num_scales,
        scale_min = args.scale_min, 
        scale_max = args.scale_max,
        metric = args.metric
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)
    model.compile(optimizer = optimizer)

    # callbacks
    callbacks = get_callbacks(
        model = model,
        val_data = val_ds,
        save_dir = args.save_dir,
        save_every_epochs = args.save_every_epochs,
        early_stopping_patience = args.early_stopping_patience,
        log_images_every = args.log_images_every,
        num_images_to_log = args.num_images_to_log,
    )

    # train the model
    model.fit(
        train_ds,
        epochs = args.epochs,
        validation_data = val_ds,
        callbacks = callbacks,
    )

    # save the final model
    final_path = os.path.join(args.save_dir, "final_model")
    model.save(final_path)
    print(f"✅ Final model saved to: {final_path}")

    # evaluate on test set
    print("\n📊 Evaluating on test set...")
    results = model.evaluate(test_ds, return_dict=True)
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test BPP: {results['bpp']:.4f}")
    print(f"Test Distortion: {results['distortion']:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=256, help="Size of image patches for training and validation.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patches_per_image", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)

    # Model args
    parser.add_argument("--main_channels", type=int, default=192)
    parser.add_argument("--hyper_channels", type=int, default=128)

    # Training args
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_scales", type=int, default=64, help="Number of Gaussian scales to prepare range coding tables for.") # Used in calculation of offset and factor
    parser.add_argument("--scale_min", type=float, default=0.11, help="Minimum value of std. dev. of Gaussians") # Used in calculation of offset and factor
    parser.add_argument("--scale_max", type=float, default=256, help="Maximum value of std. dev. of Gaussians") # Used in calculation of offset and factor
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_val", type=float, default=0.01)
    parser.add_argument("--metric", type=str, choices=["mse", "ms-ssim"], default="mse")
    parser.add_argument("--save_dir", type=str, default="saved_model")

    # Callbacks args
    parser.add_argument("--save_every_epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--log_images_every", type=int, default=0)
    parser.add_argument("--num_images_to_log", type=int, default=4)

    args = parser.parse_args()
    main(args)
