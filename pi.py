import time
import cv2
import numpy as np

from picamera2 import Picamera2

from hailo_platform import (
    HEF,
    VDevice,
    Device,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    InputVStreams,
    OutputVStreams,
    HailoStreamInterface,
)

HEF_PATH = "/home/pi/hailo-test/hef/shapes224.hef"

# ordine delle classi = stesso ordine delle cartelle in data224/train
CLASS_NAMES = [
    "circle",
    "square",
    "triangle",
]


def classify(output_tensor):
    """
    output_tensor shape: (3,)
    Valori 0–255 -> normalizzati in [0,1] con /255.0
    """
    output_tensor = np.array(output_tensor).reshape(-1).astype(np.float32)
    probs = output_tensor / 255.0

    class_id = int(np.argmax(probs))
    score = float(probs[class_id])

    if 0 <= class_id < len(CLASS_NAMES):
        label = CLASS_NAMES[class_id]
    else:
        label = f"id_{class_id}"

    return class_id, label, score, probs


def main():
    # --- Modello Hailo ---
    hef = HEF(HEF_PATH)
    input_info = hef.get_input_vstream_infos()[0]
    in_h, in_w = input_info.shape[0], input_info.shape[1]
    print("Input vstream info:", input_info)
    print(f"Input model size: {in_w}x{in_h}")

    # --- Camera ---
    picam = Picamera2()
    video_config = picam.create_video_configuration(
        main={"size": (in_w, in_h), "format": "RGB888"}
    )
    picam.configure(video_config)
    picam.start()
    time.sleep(0.5)

    # --- Dispositivi Hailo ---
    devices = Device.scan()
    if not devices:
        raise RuntimeError("Nessun dispositivo Hailo trovato")
    print("Dispositivi Hailo:", devices)

    # --- VDevice + network group ---
    with VDevice(device_ids=devices) as target:
        cfg_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )
        network_groups = target.configure(hef, cfg_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group)

        with InputVStreams(network_group, input_vstreams_params) as in_streams, \
             OutputVStreams(network_group, output_vstreams_params) as out_streams, \
             network_group.activate(network_group_params):

            print("Classificazione avviata. Premi 'q' per uscire.")

            while True:
                # frame RGB uint8 0–255
                frame_rgb = picam.capture_array()

                # niente /255 qui: lo fa il layer Rescaling nel modello
                network_input = frame_rgb.astype(np.uint8)
                input_tensor = np.expand_dims(network_input, axis=0)

                # invio all'inference
                for s in in_streams:
                    s.send(input_tensor)

                # lettura output
                outputs = []
                for s in out_streams:
                    outputs.append(s.recv())

                logits = outputs[0]       # shape (3,)

                class_id, label, score, probs = classify(logits)

                # visualizzazione
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                text_main = f"Pred: {label} ({class_id})  {score:.2f}"
                cv2.putText(
                    frame_bgr,
                    text_main,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                for i, p in enumerate(probs):
                    cname = CLASS_NAMES[i]
                    line = f"{i}: {cname} -> {p:.2f}"
                    cv2.putText(
                        frame_bgr,
                        line,
                        (10, 60 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                cv2.imshow("Circle / Square / Triangle - Hailo", frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    picam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

