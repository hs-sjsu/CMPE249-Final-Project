import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
from .paths import ROOT
from .data.kitti import load_frame
def environment():
    import torch
    print(f'Python: {sys.executable}')
    print(f'PyTorch: {torch.__version__}')
    print(f'ROCm/HIP: {torch.version.hip}')
    print(f'GPU available: {torch.cuda.is_available()}')
    if torch.version.hip is None:
        raise RuntimeError('The imported PyTorch build does not include ROCm/HIP support')
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm PyTorch is loaded, but no usable GPU is visible')
    device = torch.device('cuda:0')
    properties = torch.cuda.get_device_properties(device)
    print(f'GPU: {properties.name}')
    print(f'GPU memory: {properties.total_memory / 2 ** 30:.2f} GiB')
    torch.set_float32_matmul_precision('highest')
    generator = torch.Generator(device='cpu').manual_seed(0)
    a = torch.randn(256, 256, generator=generator)
    b = torch.randn(256, 256, generator=generator)
    expected = a @ b
    a_gpu = a.to(device)
    b_gpu = b.to(device)
    output = a_gpu @ b_gpu
    torch.cuda.synchronize(device)
    received = output.cpu()
    if not torch.isfinite(received).all().item():
        raise RuntimeError('GPU calculation produced NaN or infinity')
    torch.testing.assert_close(received, expected, rtol=0.001, atol=0.001)
    maximum_error = (received - expected).abs().max().item()
    print(f'Result device: {output.device}')
    print(f'Maximum absolute CPU/GPU difference: {maximum_error:.6g}')
    print('PASS: GPU execution succeeded and agrees with the CPU reference.')
def repeatability(argv=None):
    import torch
    import transformers
    from . import models as depth_model
    from .models import MetricDepthEstimator, MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--model-size', choices=tuple(MODELS), default='small')
    args = parser.parse_args(argv)
    frame = load_frame(args.frame)
    print(f'Model code: {depth_model.__file__}', flush=True)
    print(f'PyTorch: {torch.__version__}; HIP: {torch.version.hip}', flush=True)
    print(f'Transformers: {transformers.__version__}', flush=True)
    model_id, revision = MODELS[args.model_size]
    print(f'Checkpoint: {model_id}@{revision}', flush=True)
    print(f'RGB SHA256: {hashlib.sha256(frame.image.tobytes()).hexdigest()}', flush=True)
    print('SDPA policy: math_only', flush=True)
    estimator = MetricDepthEstimator(args.model_size)
    training = [name for name, module in estimator.model.named_modules() if module.training]
    print(f'Modules still in training mode: {training}', flush=True)
    if training:
        raise RuntimeError('Every model module should be in evaluation mode')
    backbone = getattr(estimator.model, 'backbone', None)
    config = getattr(backbone, 'config', None)
    print(f"Backbone attention: {getattr(config, '_attn_implementation', 'unknown')}", flush=True)
    directory = ROOT / 'runs' / 'repeatability_math' / args.model_size
    directory.mkdir(parents=True, exist_ok=True)
    first = None
    stable = True
    for i in range(3):
        depth = estimator.predict(frame.image.copy())
        if first is None:
            first = depth.copy()
        difference = np.abs(depth - first)
        close = np.allclose(depth, first, rtol=0.0001, atol=0.0001)
        stable = stable and close
        np.save(directory / f'{frame.frame_id}_run{i + 1}.npy', depth)
        print(f'Run {i + 1}: min={depth.min():.4f}, median={np.median(depth):.4f}, max={depth.max():.4f}, std={depth.std():.4f} m; max difference from run 1={difference.max():.6g} m; close={close}', flush=True)
    print(f'Network input shape: {estimator.input_shape}', flush=True)
    if not stable:
        raise RuntimeError('Repeated predictions disagree; inference still needs investigation')
    print('PASS: fixed-input predictions are repeatable. Accuracy still needs evaluation.')

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("unit", "environment", "repeatability", "ros"))
    args, rest = parser.parse_known_args(argv)
    if args.kind == "unit":
        from .selftest import unit_checks
        unit_checks()
    elif args.kind == "environment":
        environment()
    elif args.kind == "repeatability":
        repeatability(rest)
    else:
        from .transport_check import transport
        transport(rest)
