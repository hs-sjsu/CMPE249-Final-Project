from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from ..paths import KITTI_DIR as DATASET_DIR
CALIBRATION_SHAPES = {'P2': (3, 4), 'R0_rect': (3, 3), 'Tr_velo_to_cam': (3, 4)}

@dataclass
class KittiFrame:
    frame_id: str
    image: np.ndarray
    lidar: np.ndarray
    calibration: dict[str, np.ndarray]

def read_calibration(path):
    matrices = {}
    for line in path.read_text().splitlines():
        key, separator, text = line.partition(':')
        key = key.strip()
        if not separator or key not in CALIBRATION_SHAPES:
            continue
        if key in matrices:
            raise ValueError(f'Duplicate calibration key: {key}')
        values = np.array(text.split(), dtype=np.float64)
        shape = CALIBRATION_SHAPES[key]
        if values.size != np.prod(shape) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {key} matrix in {path}')
        matrices[key] = values.reshape(shape)
    missing = set(CALIBRATION_SHAPES) - set(matrices)
    if missing:
        raise ValueError(f'Missing calibration keys: {sorted(missing)}')
    return matrices

def load_frame(frame_number=0, dataset_dir=DATASET_DIR):
    if not 0 <= frame_number <= 999999:
        raise ValueError('Frame number must be between 0 and 999999')
    frame_id = f'{frame_number:06d}'
    training = Path(dataset_dir).resolve() / 'training'
    image_path = training / 'image_2' / f'{frame_id}.png'
    lidar_path = training / 'velodyne' / f'{frame_id}.bin'
    calibration_path = training / 'calib' / f'{frame_id}.txt'
    for path in (image_path, lidar_path, calibration_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input file: {path}')
    with Image.open(image_path) as source:
        image = np.array(source.convert('RGB'), dtype=np.uint8)
    byte_count = lidar_path.stat().st_size
    if byte_count == 0 or byte_count % 16 != 0:
        raise ValueError(f'Invalid LiDAR file size: {lidar_path}')
    lidar = np.fromfile(lidar_path, dtype='<f4').reshape(-1, 4)
    if not np.isfinite(lidar).all():
        raise ValueError(f'LiDAR file contains nonfinite values: {lidar_path}')
    calibration = read_calibration(calibration_path)
    return KittiFrame(frame_id, image, lidar, calibration)
def project_lidar(frame):
    """Return an HxW float32 camera-depth map in meters; 0 means no return.

    Uses KITTI's P2 @ R0_rect @ Tr_velo_to_cam projection chain.
    Depth is the rectified left-camera z, including P2's camera offset.
    Pixel coordinates are zero-based and rounded with numpy.rint.
    At pixel collisions, the nearest positive depth wins.
    """
    height, width = frame.image.shape[:2]
    p2 = frame.calibration['P2']
    if not np.allclose(p2[2, :3], [0.0, 0.0, 1.0]):
        raise ValueError("Expected KITTI's normalized P2 projection matrix")
    transform = np.eye(4)
    transform[:3, :] = frame.calibration['Tr_velo_to_cam']
    rectification = np.eye(4)
    rectification[:3, :3] = frame.calibration['R0_rect']
    xyz = frame.lidar[:, :3].astype(np.float64)
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    camera = (rectification @ transform @ homogeneous.T).T
    projected = (p2 @ camera.T).T
    valid = np.isfinite(projected).all(axis=1) & (projected[:, 2] > 0)
    projected = projected[valid]
    depth = projected[:, 2]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        pixels = np.rint(projected[:, :2] / depth[:, None])
    inside = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    u, v = pixels[inside].astype(np.int64).T
    depth = depth[inside]
    sparse = np.full((height, width), np.inf, dtype=np.float64)
    np.minimum.at(sparse, (v, u), depth)
    sparse[~np.isfinite(sparse)] = 0
    return sparse.astype(np.float32)
