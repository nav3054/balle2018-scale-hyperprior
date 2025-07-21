import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob

# To only keep last 5 checkpoints
class KeepLastNCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, prefix="checkpoint_epoch_", keep_last=5):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix
        self.keep_last = keep_last

    def on_epoch_end(self, epoch, logs=None):
        # Find all checkpoint files in the directory
        pattern = os.path.join(self.checkpoint_dir, f"{self.prefix}*.h5")
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
        # If more than keep_last, delete oldest
        if len(checkpoints) > self.keep_last:
            for ckpt in checkpoints[:-self.keep_last]:
                try:
                    os.remove(ckpt)
                    print(f"Deleted old checkpoint: {ckpt}")
                except Exception as e:
                    print(f"Error deleting checkpoint {ckpt}: {e}")


# Image logger - to print orig vs reconstructed images during training
class ImageLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, save_dir, log_every=5, num_images=4):
        super().__init__()
        self.val_data = val_data
        self.save_dir = save_dir
        self.log_every = log_every
        self.num_images = num_images
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.log_every <= 0:
            return
        if (epoch + 1) % self.log_every == 0:
            for batch in self.val_data.take(1):
                x = batch
            x_hat, _, _ = self.model(x, training=False)

            # Prepare combined image grid with original and reconstructed imgs
            combined_image = self._combine_orig_recon_images(x, x_hat, self.num_images)

            file_path = os.path.join(self.save_dir, f"reconstructions_epoch_{epoch+1}.png")
            plt.imsave(file_path, combined_image)
            print(f"Saved original vs reconstructed comparison at epoch {epoch+1} to {file_path}")

    def _combine_orig_recon_images(self, originals, reconstructions, num_images):
        originals = originals[:num_images].numpy()
        reconstructions = reconstructions[:num_images].numpy()

        # normalization - images are in [0,1]
        originals = np.clip(originals, 0, 1)
        reconstructions = np.clip(reconstructions, 0, 1)

        # original img on top, reconstructed img below
        pairs = []
        for i in range(num_images):
            orig = originals[i]
            recon = reconstructions[i]

            # add orig and reconstructed labels for images
            orig = self._add_label(orig, "Original")
            recon = self._add_label(recon, "Reconstructed")

            pair = np.vstack([orig, recon])
            pairs.append(pair)

        # Concatenate pairs horizontally
        combined = np.hstack(pairs)

        # If images have channel last, matplotlib expects H x W x C
        return combined

    def _add_label(self, image, label):
        import matplotlib.patches as patches
    
        fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
        ax.imshow(image)
        ax.axis('off')
        ax.text(5, 15, label, color='white', fontsize=7, weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=2))
        
        fig.canvas.draw()
        
        img_with_label = np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), dtype=np.uint8)
        img_with_label = img_with_label.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_with_label = img_with_label[:, :, :3] 
        plt.close(fig)
    
        img_with_label = img_with_label.astype(np.float32) / 255.0
        return img_with_label



def get_callbacks(model, val_data, save_dir,
                  save_every_epochs=5,
                  early_stopping_patience=10,
                  log_images_every=5,
                  num_images_to_log=4):
    callbacks = []

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        verbose=1,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)

    # Terminate training on NaN loss
    from tensorflow.keras.callbacks import TerminateOnNaN
    terminate_on_nan = TerminateOnNaN()
    callbacks.append(terminate_on_nan)

    # comment out old ModelCheckpoint code
    # checkpoint_path = os.path.join(save_dir, 'checkpoint_epoch_{epoch:02d}.h5')
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     save_weights_only=True,
    #     save_freq='epoch',
    #     period=save_every_epochs,
    #     verbose=1,
    # )
    # callbacks.append(model_checkpoint)

    # ModelCheckpoint - save best only
    best_checkpoint_path = os.path.join(save_dir, 'checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5')
    best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1,
    )
    callbacks.append(best_checkpoint)

    # keep-last-N callback
    keep_last_n_ckpt = KeepLastNCheckpoints(
        checkpoint_dir=save_dir,
        prefix="checkpoint_epoch_",
        keep_last=5
    )
    callbacks.append(keep_last_n_ckpt)

    # Image logging callback
    image_logger = ImageLoggingCallback(
        val_data=val_data,
        save_dir=save_dir,
        log_every=log_images_every,
        num_images=num_images_to_log
    )
    callbacks.append(image_logger)

    return callbacks
