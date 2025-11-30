import os
import argparse
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparam
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # Dati
    parser.add_argument("--data_dir", type=str, default="data224")
    parser.add_argument("--img_size", type=int, default=224)

    # Output modello
    # Su SageMaker questo verrà sovrascritto da SM_MODEL_DIR
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "model"),
    )

    # Opzionale: esportare anche il .tflite
    parser.add_argument("--export_tflite", action="store_true")

    args = parser.parse_args()
    return args


def get_datasets(data_dir, img_size, batch_size):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    img_size_tuple = (img_size, img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size_tuple,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size_tuple,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    print("Classi trovate:", class_names)

    # Performance: cache + prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, len(class_names)


def build_model(img_size, num_classes, learning_rate):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255, input_shape=(img_size, img_size, 3)),

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def main():
    args = parse_args()

    print("Argomenti:", args)

    train_ds, val_ds, num_classes = get_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    model = build_model(
        img_size=args.img_size,
        num_classes=num_classes,
        learning_rate=args.learning_rate,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
    )

    # Salvataggio SavedModel (SageMaker usa SM_MODEL_DIR)
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # opzionale: versione "1" come best practice per SageMaker
    export_path = os.path.join(model_dir, "1")
    print(f"Salvo SavedModel in: {export_path}")
    model.export(export_path)

    if args.export_tflite:
        # Conversione a TFLite
        print("Converto a TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path = os.path.join(model_dir, "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"Salvato modello TFLite in: {tflite_path}")
        print("Dimensione TFLite (byte):", len(tflite_model))


if __name__ == "__main__":
    main()

