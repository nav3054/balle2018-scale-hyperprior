import os
import tarfile
import tempfile
import tensorflow as tf
import random

SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def unzip_tar_file(tar_path, extract_dir):
    if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
        print(f"Extracting {tar_path} to {extract_dir}...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_dir)
        print("Extraction complete.")
    else:
        print(f"Using existing extracted directory: {extract_dir}")
    return extract_dir


def list_images(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTS):
                all_files.append(os.path.join(root, file))
    return all_files


def build_patch_dataset(image_paths, patch_size, batch_size, patches_per_image):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def decode_and_crop_multiple(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.convert_image_dtype(image, tf.float32)
        shape = tf.shape(image)
        height, width = shape[0], shape[1]

        def crop_patch(_):
            return tf.image.random_crop(image, [patch_size, patch_size, 3])

        def crop_many():
            patches = tf.data.Dataset.range(patches_per_image)
            patches = patches.map(crop_patch, num_parallel_calls=tf.data.AUTOTUNE)
            return patches

        def skip_image():
            return tf.data.Dataset.from_tensors(tf.zeros([patch_size, patch_size, 3], tf.float32))

        return tf.cond(
            tf.logical_or(height < patch_size, width < patch_size),
            true_fn=skip_image,
            false_fn=crop_many
        )

    dataset = dataset.interleave(
        decode_and_crop_multiple,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    dataset = dataset.filter(lambda x: tf.reduce_sum(x) > 0.0)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_train_val_test_datasets(
    input_path,
    patch_size,
    batch_size,
    patches_per_image=1,
    val_split=0.1,
    test_split=0.1,
    max_images=None,
    seed=0
):
    if input_path.endswith(".tar"):
        extract_dir = os.path.join(tempfile.gettempdir(), "imagenet_val")
        dataset_dir = unzip_tar_file(input_path, extract_dir)
    else:
        dataset_dir = input_path

    image_paths = list_images(dataset_dir)
    print(f"Total images found: {len(image_paths)}")

    if max_images is not None:
        image_paths = image_paths[:max_images]
        print(f"Limiting to first {max_images} images.")

    random.seed(seed)
    random.shuffle(image_paths)

    total = len(image_paths)
    val_size = int(total * val_split)
    test_size = int(total * test_split)
    train_size = total - val_size - test_size

    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:train_size + val_size]
    test_paths = image_paths[train_size + val_size:]

    print(f"Splitting dataset → Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    train_ds = build_patch_dataset(train_paths, patch_size, batch_size, patches_per_image)
    val_ds = build_patch_dataset(val_paths, patch_size, batch_size, patches_per_image)
    test_ds = build_patch_dataset(test_paths, patch_size, batch_size, patches_per_image)

    return train_ds, val_ds, test_ds
