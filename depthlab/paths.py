import os
from pathlib import Path
ROOT = Path(os.environ.get("DEPTHLAB_ROOT", Path.cwd())).expanduser().resolve()
KITTI_DIR = ROOT / "dataset"
VKITTI_DIR = KITTI_DIR / "virtual_kitti2"
