import tensorflow as tf
import tensorflow_compression as tfc
from training.losses import compute_rate_distortion_loss

class Balle2018Hyperprior(tf.keras.Model):
    def __init__(self, main_channels=192, hyper_channels=128, lambda_rd=0.01, metric='mse'):
        super().__init__()

        num_scales = 64
        downsample_factor = 16

        self.main_channels = main_channels
        self.hyper_channels = hyper_channels
        self.downsample_factor = downsample_factor

        self.lambda_rd = lambda_rd
        self.metric = metric

        # MAIN ENCODER (g_a)
        self.main_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(main_channels, 5, strides=2, padding='same'),
            tfc.layers.GDN(),
            tf.keras.layers.Conv2D(main_channels, 5, strides=2, padding='same'),
            tfc.layers.GDN(),
            tf.keras.layers.Conv2D(main_channels, 5, strides=2, padding='same'),
            tfc.layers.GDN(),
            tf.keras.layers.Conv2D(main_channels, 5, strides=2, padding='same'),
        ])

        # HYPER ENCODER (h_a)
        self.hyp_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hyper_channels, 3, strides=1, padding='same', activation=tf.nn.relu),
            tf.keras.layers.Conv2D(hyper_channels, 5, strides=2, padding='same', activation=tf.nn.relu),
            tf.keras.layers.Conv2D(hyper_channels, 5, strides=2, padding='same', activation=None),
        ])

        # HYPER DECODER (h_s)
        self.hyp_decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(hyper_channels, 5, strides=2, padding='same', activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(hyper_channels, 5, strides=2, padding='same', activation=tf.nn.relu),
            tf.keras.layers.Conv2D(main_channels * 2, 3, strides=1, padding='same', activation=None),
        ])

        # MAIN DECODER (g_s)
        self.main_decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(main_channels, 5, strides=2, padding='same'),
            tfc.layers.GDN(inverse=True),
            tf.keras.layers.Conv2DTranspose(main_channels, 5, strides=2, padding='same'),
            tfc.layers.GDN(inverse=True),
            tf.keras.layers.Conv2DTranspose(main_channels, 5, strides=2, padding='same'),
            tfc.layers.GDN(inverse=True),
            tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding='same', activation=None),
        ])

        # trainable scale function params
        self.offset = tf.Variable(-1.5, dtype=tf.float32, trainable=True, name="scale_offset")
        self.factor = tf.Variable(0.175, dtype=tf.float32, trainable=True, name="scale_factor")

        
        # ENTROPY MODELS
        self.main_entropy_model = tfc.LocationScaleIndexedEntropyModel(
            prior_fn=tfc.NoisyNormal(),
            num_scales=num_scales,
            scale_fn=lambda i: tf.exp(self.offset + self.factor * tf.cast(i, tf.float32)),
            coding_rank=3,
            compression=True
        )

        self.hyperprior_entropy_model = tfc.ContinuousBatchedEntropyModel(
            prior_fn=tfc.NoisyDeepFactorized(),
            coding_rank=3,
            compression=True
        )

  
    @tf.function
    def call(self, x, training=False):
        y = self.main_encoder(x)
        z = self.hyp_encoder(tf.abs(y))
        z_hat, z_likelihoods = self.hyperprior_entropy_model(z, training=training)

        hyper_output = self.hyp_decoder(z_hat)
        loc, scale = tf.split(hyper_output, num_or_size_splits=2, axis=-1)

        y_hat, y_likelihoods = self.main_entropy_model(y, indexes=scale, means=loc, training=training)
        x_hat = self.main_decoder(y_hat)

        return x_hat, y_likelihoods, z_likelihoods

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x_hat, y_likelihoods, z_likelihoods = self(x, training=True)
            loss, (bpp, distortion) = compute_rate_distortion_loss(
                x, x_hat, y_likelihoods, z_likelihoods,
                lambda_rd=self.lambda_rd,
                metric=self.metric,
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss, "bpp": bpp, "distortion": distortion}

    def test_step(self, data):
        x = data
        x_hat, y_likelihoods, z_likelihoods = self(x, training=False)
        loss, (bpp, distortion) = compute_rate_distortion_loss(
            x, x_hat, y_likelihoods, z_likelihoods,
            lambda_rd=self.lambda_rd,
            metric=self.metric,
        )
        return {"loss": loss, "bpp": bpp, "distortion": distortion}



    # compress image function
    @tf.function
    def compress(self, x):
        y = self.main_encoder(x)
        z = self.hyp_encoder(tf.abs(y))
        z_hat, z_strings = self.hyperprior_entropy_model.compress(z)

        hyper_output = self.hyp_decoder(z_hat)
        loc, scale = tf.split(hyper_output, num_or_size_splits=2, axis=-1)
        y_hat, y_strings = self.main_entropy_model.compress(y, indexes=scale, means=loc)

        return {"strings": [y_strings, z_strings], "shape": tf.shape(x)[1:-1]}


    # decompress image function
    @tf.function
    def decompress(self, strings, shape):
        y_strings, z_strings = strings
        z_shape = (
            shape[0] // self.downsample_factor,
            shape[1] // self.downsample_factor,
            self.hyp_encoder.layers[-1].filters
        )

        z_hat = self.hyperprior_entropy_model.decompress(z_strings, shape=z_shape)
        hyper_output = self.hyp_decoder(z_hat)
        loc, scale = tf.split(hyper_output, num_or_size_splits=2, axis=-1)

        y_hat = self.main_entropy_model.decompress(y_strings, indexes=scale, means=loc)
        x_hat = self.main_decoder(y_hat)
        return x_hat
