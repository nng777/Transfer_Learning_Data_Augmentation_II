#!/usr/bin/env python3
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse
import sys
from textwrap import dedent
from typing import Optional


def explain() -> None:
    message = dedent(
        """
        Improving Your Image Models: Transfer Learning and Data Augmentation with Keras
        -----------------------------------------------------------------------------
        What you'll learn in a minute:
          1) Why transfer learning helps: reuse a pretrained CNN (e.g., MobileNetV2) as a feature extractor so you need less data and train faster.
          2) How augmentation reduces overfitting: random flips/rotations/color jitter create new variations from your images.
          3) Typical workflow in Keras:
             - Load a base model with pretrained weights
             - Freeze base, add a small classification head, train quickly
             - Unfreeze top layers and fine-tune with a low learning rate
             - Use tf.keras.layers for augmentation in the input pipeline

        Try running with --demo-augment or --demo-train (requires TensorFlow installed).
        """
    ).strip()
    print(message)


def demo_augment(sample_count: int = 4) -> None:
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as exc:
        print("TensorFlow (and NumPy) is required for this demo. See requirements.txt.")
        print(f"Import error: {exc}")
        sys.exit(2)

    rng = np.random.default_rng(42)
    sample = (rng.random((128, 128, 3)) * 255).astype("uint8")
    sample = tf.convert_to_tensor(sample)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    print("Applying augmentation to a synthetic 128x128 RGB image...")
    for i in range(sample_count):
        aug = data_augmentation(tf.expand_dims(sample, 0), training=True)
        arr = tf.cast(tf.squeeze(aug, 0), tf.uint8).numpy()
        # Save to outputs for quick visual inspection
        from PIL import Image
        img = Image.fromarray(arr)
        out_path = f"outputs/augmented_{i+1}.png"
        img.save(out_path)
        print(f"Saved {out_path}")



def demo_train(epochs: int = 1, image_size: int = 160, data_dir: Optional[str] = None) -> None:
    try:
        import tensorflow as tf
    except Exception as exc:
        print("TensorFlow is required for this demo. See requirements.txt.")
        print(f"Import error: {exc}")
        sys.exit(2)

    tf.get_logger().setLevel("ERROR")

    import os
    # Resolve dataset directory
    if data_dir is not None:
        base_dir = data_dir
    else:
        # Small built-in dataset: cats_vs_dogs subset via keras.utils.get_file
        url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
        path_to_zip = tf.keras.utils.get_file(
            "cats_and_dogs_filtered.zip", origin=url, extract=True
        )
        cache_dir = os.path.dirname(path_to_zip)
        base_dir = os.path.join(cache_dir, "cats_and_dogs_filtered")

        # If expected directories are missing, attempt manual extraction and auto-discovery
        expected_train = os.path.join(base_dir, "train")
        expected_val = os.path.join(base_dir, "validation")
        if not (os.path.isdir(expected_train) and os.path.isdir(expected_val)):
            try:
                import zipfile

                with zipfile.ZipFile(path_to_zip, "r") as zf:
                    zf.extractall(cache_dir)
            except Exception as exc:
                print("Failed to extract dataset zip. You can pass --data-dir to use a local dataset.")
                print(f"Extraction error: {exc}")
                sys.exit(2)

            # Try to discover the extracted folder that contains 'train' and 'validation'
            discovered = None
            for name in os.listdir(cache_dir):
                candidate = os.path.join(cache_dir, name)
                if os.path.isdir(os.path.join(candidate, "train")) and os.path.isdir(
                    os.path.join(candidate, "validation")
                ):
                    discovered = candidate
                    break
            if discovered is not None:
                base_dir = discovered
            else:
                print(
                    "Could not locate 'train' and 'validation' after extraction. "
                    "Pass --data-dir pointing to a directory with those subfolders."
                )
                sys.exit(2)

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "validation")

    batch_size = 16
    img_size = (image_size, image_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size
    )

    AUTOTUNE = tf.data.AUTOTUNE
    class_names = train_ds.class_names

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    print("Stage 1: Train classification head with base frozen...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    print("Stage 2: Fine-tune top layers with low LR...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=max(1, epochs // 2))

    model.save("outputs/transfer_learning_model")
    print("Saved model to outputs/transfer_learning_model")



def main():
    parser = argparse.ArgumentParser(description="Keras Transfer Learning & Augmentation Demo")
    parser.add_argument("mode", choices=["explain", "demo-augment", "demo-train"], help="What to run")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for demo-train")
    parser.add_argument("--image-size", type=int, default=160, help="Image size for demo-train")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Optional: path to dataset directory containing 'train' and 'validation' subfolders; "
            "if omitted, the script downloads a small cats_vs_dogs subset."
        ),
    )
    parser.add_argument("--samples", type=int, default=4, help="Number of augmented samples to save")
    args = parser.parse_args()

    if args.mode == "explain":
        explain()
    elif args.mode == "demo-augment":
        demo_augment(args.samples)
    elif args.mode == "demo-train":
        demo_train(args.epochs, args.image_size, args.data_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
