import numpy as np
from PIL import Image
from ..paths import VKITTI_DIR as DATASET
SCENES = ('Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20')

def read_pair(scene, frame):
    if scene not in SCENES or not 0 <= frame <= 99999:
        raise ValueError('Invalid scene or frame number')
    base = DATASET / scene / 'clone' / 'frames'
    with Image.open(base / 'rgb/Camera_0' / f'rgb_{frame:05d}.jpg') as image:
        rgb = np.array(image.convert('RGB'), dtype=np.uint8)
    with Image.open(base / 'depth/Camera_0' / f'depth_{frame:05d}.png') as image:
        raw = np.array(image)
    if raw.ndim != 2 or raw.dtype.kind not in 'iu' or raw.dtype.itemsize < 2:
        raise ValueError('Expected a single-channel 16-bit depth PNG')
    if raw.min() < 0 or raw.max() > 65535 or raw.shape != rgb.shape[:2]:
        raise ValueError('Invalid depth values or mismatched RGB/depth dimensions')
    return (rgb, raw.astype(np.float32) / 100.0)
