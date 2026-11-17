from pathlib import Path
import hashlib
import sys
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from .paths import ROOT
CACHE_DIR = ROOT / "checkpoints/huggingface"
MODELS = {'small': ('depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf', 'fd2c22027eaf20374204f14099b8341e1925ad39'), 'large': ('depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf', '4eab4cf1983c2801c515804005214de56a4b67cc')}

class MetricDepthEstimator:

    def __init__(self, model_size='small'):
        if model_size not in MODELS:
            raise ValueError("model_size must be 'small' or 'large'")
        self.model_size = model_size
        self.model_id, self.revision = MODELS[model_size]
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError('A working ROCm PyTorch GPU environment is required')
        self.device = torch.device('cuda:0')
        torch.set_float32_matmul_precision('highest')
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        options = {'revision': self.revision, 'cache_dir': str(CACHE_DIR)}
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=False, **options)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id, use_safetensors=True, **options).to(device=self.device, dtype=torch.float32).eval()
        if self.model.config.depth_estimation_type != 'metric':
            raise ValueError('Expected a metric-depth checkpoint')
        self.input_shape = None

    def metadata(self):
        return {'model_size': self.model_size, 'model': self.model_id, 'revision': self.revision, 'precision': str(next(self.model.parameters()).dtype), 'sdpa_backend_policy': 'math_only', 'device': str(self.device), 'gpu': torch.cuda.get_device_name(0), 'pytorch': str(torch.__version__), 'hip': torch.version.hip, 'transformers': transformers.__version__, 'python': sys.version, 'python_executable': sys.executable, 'depth_model_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    @torch.inference_mode()
    def predict(self, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected an HxWx3 RGB uint8 image')
        inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
        self.input_shape = tuple(inputs['pixel_values'].shape)
        with sdpa_kernel(SDPBackend.MATH):
            outputs = self.model(**inputs)
        resized = F.interpolate(outputs.predicted_depth.unsqueeze(1).float(), size=rgb.shape[:2], mode='bicubic', align_corners=False)[0, 0]
        depth = resized.cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError('Model returned nonfinite or nonpositive depth')
        return depth
