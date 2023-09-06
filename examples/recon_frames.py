#
# you need to export PYTHONPATH=/build_directory:${PYTHONPATH}
# such that it finds the _simple_image_recon python library
#

import numpy as np
from pathlib import Path
from _simple_image_recon import SimpleImageRecon

from numpy.lib.recfunctions import unstructured_to_structured
import cv2
import os

# from matplotlib import pyplot as plt


width = 970  # sensor resolution
height = 625  # sensor resolution
T_cut = 30
tile_size = 2  # 2x2 tile size for activity detection
fill_ratio = 0.6  # average fill ratio to maintain for tiles

recon = SimpleImageRecon(width, height, T_cut, 2, fill_ratio)
base = "../claude/seq_ball"
output_base = "./frames"
if not os.path.exists(str(output_base)):
    os.makedirs(str(output_base))

num_files = 0
for i in range(100000):
    ev_file = Path(base + f"/events_{i:06d}.npy")
    if ev_file.is_file():
        num_files += 1
        # rearrange columns to match structured layout
        float_events = np.load(ev_file)[:, (0, 1, 3, 2)]
        # filter out bad range events
        fe = float_events[(float_events[:, 0] < width) & (float_events[:, 1] < height), :]
        # convert to structured array like metavision SDK
        events = unstructured_to_structured(
            fe,
            dtype=np.dtype([("x", "u2"), ("y", "u2"), ("p", "i1"), ("t", "i4")]),
        )
        recon.update(events)
        out_file = Path(output_base + f"/frame_{i:06d}.png")
        img = recon.get_state()["L"]  # L has the reconstructed brightness
        img_n = cv2.normalize(
            img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        print(
            f"frame {i:4d} has {events.shape[0]:6} events and range {img.min():.2f} -> {img.max():.2f}"
        )
        cv2.imwrite(str(out_file), img_n)
        # plt.imshow(img, cmap="gray")
        # plt.show()
    else:
        print(f"created {num_files} frames")
        break
