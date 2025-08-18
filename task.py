import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10


def load_data():
    """Load and preprocess the CIFAR-10 dataset."""
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def build_model(num_classes: int = 10):
    """Create a MobileNetV2-based model for CIFAR-10."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
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
    """Train only the classification head and return validation accuracy."""
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
    """Unfreeze top layers of the base model and continue training."""
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
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
    )
    return history.history["val_accuracy"][-1]


def main() -> None:
    (train_images, train_labels), (test_images, test_labels) = load_data()
    # Use a subset for speed during demos
    train_images = train_images[:10000]
    train_labels = train_labels[:10000]
    test_images = test_images[:2000]
    test_labels = test_labels[:2000]

    model, base_model = build_model(num_classes=10)
    val_acc_initial = initial_training(
        model,
        (train_images, train_labels),
        (test_images, test_labels),
        epochs=1,
    )
    val_acc_finetune = fine_tune(
        model,
        base_model,
        (train_images, train_labels),
        (test_images, test_labels),
        epochs=1,
    )

    improvement = val_acc_finetune - val_acc_initial
    print(
        f"Validation accuracy improved from {val_acc_initial:.4f} to {val_acc_finetune:.4f} ({improvement:+.4f})."
    )


if __name__ == "__main__":
    main()
