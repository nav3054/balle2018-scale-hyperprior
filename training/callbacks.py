import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


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

            # Prepare combined image grid with original and reconstructed
            combined_image = self._combine_orig_recon_images(x, x_hat, self.num_images)

            file_path = os.path.join(self.save_dir, f"reconstructions_epoch_{epoch+1}.png")
            plt.imsave(file_path, combined_image)
            print(f"Saved original vs reconstructed comparison at epoch {epoch+1} to {file_path}")

    def _combine_orig_recon_images(self, originals, reconstructions, num_images):
        originals = originals[:num_images].numpy()
        reconstructions = reconstructions[:num_images].numpy()

        # Assume images are in [0,1]
        originals = np.clip(originals, 0, 1)
        reconstructions = np.clip(reconstructions, 0, 1)

        # Create a single image stacking originals and reconstructions vertically
        # For each pair, stack original on top, reconstruction below
        pairs = []
        for i in range(num_images):
            orig = originals[i]
            recon = reconstructions[i]

            # Add labels
            orig = self._add_label(orig, "Original")
            recon = self._add_label(recon, "Reconstructed")

            pair = np.vstack([orig, recon])
            pairs.append(pair)

        # Concatenate pairs horizontally
        combined = np.hstack(pairs)

        # If images have channel last, matplotlib expects H x W x C
        return combined
    '''
    def _add_label(self, image, label):
        # Add label text on top of the image using matplotlib
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
        ax.imshow(image)
        ax.axis('off')
        ax.text(5, 15, label, color='white', fontsize=12, weight='bold',
                bbox=dict(facecolor='black', alpha=0.7, pad=2))
        fig.canvas.draw()

        # Convert plot to numpy array
        img_with_label = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_with_label = img_with_label.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Normalize to [0,1]
        img_with_label = img_with_label.astype(np.float32) / 255.0
        return img_with_label
    '''
    def _add_label(self, image, label):
        import matplotlib.patches as patches
    
        fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
        ax.imshow(image)
        ax.axis('off')
        ax.text(5, 15, label, color='white', fontsize=7, weight='bold',
                bbox=dict(facecolor='black', alpha=0.7, pad=2))
        
        fig.canvas.draw()
        img_with_label = np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), dtype=np.uint8)
        img_with_label = img_with_label.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_with_label = img_with_label[:, :, :3]  # Remove alpha channel if needed
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

    # Model checkpoint (saving weights every save_every_epochs)
    checkpoint_path = os.path.join(save_dir, 'checkpoint_epoch_{epoch:02d}.h5')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_freq='epoch',
        period=save_every_epochs,
        verbose=1,
    )
    callbacks.append(model_checkpoint)

    # Image logging callback
    image_logger = ImageLoggingCallback(
        val_data=val_data,
        save_dir=save_dir,
        log_every=log_images_every,
        num_images=num_images_to_log
    )
    callbacks.append(image_logger)

    return callbacks
