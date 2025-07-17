import tensorflow as tf

def compute_rate_distortion_loss(x, x_hat, y_likelihoods, z_likelihoods, lambda_rd, metric="mse"):
    """
    Computes rate-distortion loss for end-to-end image compression.

    Args:
        x: Original image tensor. Shape: [B, H, W, C]
        x_hat: Reconstructed image tensor. Shape: [B, H, W, C]
        y_likelihoods: Likelihoods from main entropy model (LocationScaleIndexedEntropyModel)
        z_likelihoods: Likelihoods from hyperprior entropy model (ContinuousBatchedEntropyModel)
        lambda_rd: Lagrange multiplier (λ) for distortion
        metric: "mse" or "ms-ssim"

    Returns:
        total_loss: λ * distortion + bpp
        (bpp, distortion): Tuple of scalars for logging
    """
    # Bits per pixel = -log2(p) / num_pixels
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[1:-1]), tf.float32)

    bpp_y = tf.reduce_sum(tf.math.log(y_likelihoods)) / (-tf.math.log(2.0) * num_pixels)
    bpp_z = tf.reduce_sum(tf.math.log(z_likelihoods)) / (-tf.math.log(2.0) * num_pixels)
    bpp = bpp_y + bpp_z

    if metric == "mse":
        distortion = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    elif metric == "ms-ssim":
        # Assumes inputs are in [0, 1]
        distortion = 1 - tf.image.ssim_multiscale(x, x_hat, max_val=1.0)
        distortion = tf.reduce_mean(distortion)
    else:
        raise ValueError(f"Unsupported distortion metric: {metric}")

    total_loss = lambda_rd * distortion + bpp
    return total_loss, (bpp, distortion)
