import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


def load_data():
    #Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def build_augmentation(
    flip: str = "horizontal", rotation: float = 0.1, zoom: float = 0.1
):
    #Return a Sequential augmentation model with common layers.
    return tf.keras.Sequential(
        [
            layers.RandomFlip(flip),
            layers.RandomRotation(rotation),
            layers.RandomZoom(zoom),
        ]
    )


def build_model(
    num_classes: int = 10, augment_layer: tf.keras.Sequential | None = None
):
    #Create a MobileNetV2-based model for CIFAR-10.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = inputs
    if augment_layer is not None:
        x = augment_layer(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, base_model


def initial_training(model, train_ds, val_ds, epochs: int = 1):
    #Train only the classification head and return validation accuracy.
    history = model.fit(
        train_ds[0],
        train_ds[1],
        batch_size=64,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
    )
    return history.history["val_accuracy"][-1]


def fine_tune(model, base_model, train_ds, val_ds, epochs: int = 1):
    #Unfreeze top layers of the base model and continue training.
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds[0],
        train_ds[1],
        batch_size=64,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
    )
    return history.history["val_accuracy"][-1]


def visualize_augmentations(
    augment_layer: tf.keras.Sequential,
    images,
    num_examples: int = 5,
    save_path: str = "outputs/augmented_examples.png",
) -> None:
    #Save a grid of original and augmented images for sanity checking.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(num_examples * 2, 4))
    for i in range(num_examples):
        augmented = augment_layer(tf.expand_dims(images[i], 0), training=True)
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(tf.clip_by_value(augmented[0], 0, 1))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved augmented examples to {save_path}")


def generate_report(
    val_acc_initial: float,
    val_acc_finetune: float,
    val_acc_aug: float,
    output_path: str = "REPORT.md",
) -> None:
    #Create a markdown report summarizing training results.
    dataset_desc = (
        "CIFAR-10 contains 60,000 32x32 colour images across 10 classes. "
        "This run uses 10,000 training and 2,000 validation examples."
    )
    improvement = val_acc_aug - val_acc_finetune
    report = f"""# Training Report

## Dataset
{dataset_desc}

## Results
- Validation accuracy after initial training: {val_acc_initial:.4f}
- Validation accuracy after fine-tuning: {val_acc_finetune:.4f}
- Validation accuracy after data augmentation: {val_acc_aug:.4f} (change {improvement:+.4f})

## Conclusion
Fine-tuning adapts pre-trained features to the CIFAR-10 task, while data augmentation
exposes the model to varied samples. Together they improve generalisation over the
baseline model.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report to {output_path}")


def main() -> None:
    (train_images, train_labels), (test_images, test_labels) = load_data()
    #Use a subset for speed during demos
    train_images = train_images[:10000]
    train_labels = train_labels[:10000]
    test_images = test_images[:2000]
    test_labels = test_labels[:2000]

    # Baseline training without augmentation
    model_base, base_base = build_model(num_classes=10)
    val_acc_initial = initial_training(
        model_base,
        (train_images, train_labels),
        (test_images, test_labels),
        epochs=1,
    )
    val_acc_finetune = fine_tune(
        model_base,
        base_base,
        (train_images, train_labels),
        (test_images, test_labels),
        epochs=1,
    )
    print(f"Baseline validation accuracy: {val_acc_finetune:.4f}")

    #Training with recommended augmentation
    augment = build_augmentation()
    model_aug, base_aug = build_model(num_classes=10, augment_layer=augment)
    initial_aug = initial_training(
        model_aug,
        (train_images, train_labels),
        (test_images, test_labels),
        epochs=1,
    )
    finetune_aug = fine_tune(
        model_aug,
        base_aug,
        (train_images, train_labels),
        (test_images, test_labels),
        epochs=1,
    )
    improvement = finetune_aug - val_acc_finetune
    print(
        f"Augmented validation accuracy: {finetune_aug:.4f} (change {improvement:+.4f})"
    )

    #Try stronger augmentation parameters/ my experiments
    augment_strong = build_augmentation(rotation=0.3, zoom=0.3)
    model_strong, base_strong = build_model(10, augment_layer=augment_strong)
    initial_strong = initial_training(
        model_strong,
        (train_images, train_labels),
        (test_images, test_labels),
        epochs=1,
    )
    finetune_strong = fine_tune(
        model_strong,
        base_strong,
        (train_images, train_labels),
        (test_images, test_labels),
        epochs=1,
    )
    print(f"Stronger augmentation validation accuracy: {finetune_strong:.4f}")

    visualize_augmentations(augment, train_images)
    generate_report(val_acc_initial, val_acc_finetune, finetune_aug)


if __name__ == "__main__":
    main()

