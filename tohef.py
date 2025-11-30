from pathlib import Path
import numpy as np
from PIL import Image

from hailo_sdk_client import ClientRunner


TFLITE_MODEL = "model/model.tflite"
CALIB_DIR = "calib_images"   # cartella con PNG/JPG 224x224
OUTPUT_HAR = "model_h8l.har"
MODEL_NAME = "shapes224"     # nome arbitrario per il modello

IMG_SIZE = (224, 224)


def load_calib_images_list(calib_dir, img_size=IMG_SIZE):
    # prendo png e jpg (non ricorsivo, o se vuoi davvero ricorsivo usa **/*.png e **/*.jpg per entrambi)
    paths = list(Path(calib_dir).glob("*.png")) + list(Path(calib_dir).glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"Nessuna immagine trovata in {calib_dir}")
    print(f"Trovate {len(paths)} immagini di calibrazione")

    images = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(img_size)

        # ⚠️ QUI LA FIX: NIENTE /255, NIENTE float32 NECESSARIO
        arr = np.array(img, dtype=np.uint8)   # valori 0–255, come li vede la camera
        images.append(arr)

    calib_array = np.stack(images, axis=0)  # shape (N, H, W, C)
    print("Shape calibrazione:", calib_array.shape, calib_array.dtype)
    return calib_array


def main():
    # 1) Crea il runner per Hailo8L
    runner = ClientRunner(hw_arch="hailo8l")

    # 2) PARSING: traduce il modello TFLite nel formato Hailo
    print(f"Traduco il modello TFLite: {TFLITE_MODEL}")
    hn, npz = runner.translate_tf_model(
        TFLITE_MODEL,
        MODEL_NAME,
    )

    # 3) OPTIMIZATION (quantizzazione & co.) con dataset di calibrazione
    print("Avvio ottimizzazione con dataset di calibrazione...")
    calib_data = load_calib_images_list(CALIB_DIR, img_size=IMG_SIZE)
    runner.optimize(calib_data)   # molte versioni accettano direttamente (N,H,W,C)

    # 4) COMPILAZIONE → HAR per hailo8l
    print("Compilo il modello per Hailo8L...")
    runner.compile()

    # 5) Salva HAR
    runner.save_har(OUTPUT_HAR)
    print(f"HAR generato: {OUTPUT_HAR}")


if __name__ == "__main__":
    main()

