import cv2
import numpy as np
from PIL import Image

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

HEF_PATH = "/home/pi/hailo-test/model_h8l.hef"
IMG_SIZE = 224
CLASS_NAMES = ["circle", "square", "triangle"]

TEST_IMAGES = [
    "data224/val/circle/circle_0071.png",
    "data224/val/square/square_0071.png",
    "data224/val/triangle/triangle_0071.png",
]


def load_img_for_hailo(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.uint8)        # ⚠️ UINT8 0–255
    x = np.expand_dims(x, axis=0)            # (1, 224, 224, 3)
    return x


def main():
    hef = HEF(HEF_PATH)
    input_info = hef.get_input_vstream_infos()[0]
    print("Input info:", input_info)

    devices = Device.scan()
    if not devices:
        raise RuntimeError("Nessun dispositivo Hailo trovato")
    print("Dispositivi:", devices)

    with VDevice(device_ids=devices) as target:
        cfg_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )
        network_groups = target.configure(hef, cfg_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        in_params = InputVStreamParams.make_from_network_group(network_group)
        out_params = OutputVStreamParams.make_from_network_group(network_group)

        with InputVStreams(network_group, in_params) as in_streams, \
             OutputVStreams(network_group, out_params) as out_streams, \
             network_group.activate(network_group_params):

            for path in TEST_IMAGES:
                x = load_img_for_hailo(path)

                for s in in_streams:
                    s.send(x)

                outputs = []
                for s in out_streams:
                    outputs.append(s.recv())

                logits = np.array(outputs[0]).reshape(-1)
                probs = logits / 255.0

                cid = int(np.argmax(probs))
                print("Hailo", path, "->", CLASS_NAMES[cid], "logits:", logits, "probs:", probs)


if __name__ == "__main__":
    main()

