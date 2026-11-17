"""Shared measurement boundaries for evaluation and controlled benchmarks."""
import time
import numpy as np

PROFILE_TOPIC = "/depth/profile"
PROTOCOL = {
    "forward_gpu_ms": "GPU events around the model forward call only",
    "predict_wall_ms": "processor + H2D + forward + resize + D2H + validation + final synchronization",
    "ros_round_trip_ms": "client before RGB message construction until matching depth is decoded",
    "request_complete_ms": "client before RGB construction until depth and matching telemetry are received and checked",
    "memory": "observed PyTorch allocator peaks; includes resident model; reserved includes cache; excludes non-PyTorch allocations",
    "instrumentation": "profiling enabled, synchronized, batch size 1, one request in flight",
}

class GpuMeasurement:
    def __init__(self, estimator):
        import torch
        self.torch = torch
        self.estimator = estimator
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.handles = [estimator.model.register_forward_pre_hook(self.mark_begin),
                        estimator.model.register_forward_hook(self.mark_end)]

    def mark_begin(self, module, inputs):
        self.begin.record()

    def mark_end(self, module, inputs, output):
        self.end.record()

    def predict(self, rgb):
        cuda = self.torch.cuda
        device = self.estimator.device
        cuda.synchronize(device)
        cuda.reset_peak_memory_stats(device)
        started = time.perf_counter_ns()
        depth = self.estimator.predict(rgb)
        cuda.synchronize(device)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        metrics = {
            "forward_gpu_ms": float(self.begin.elapsed_time(self.end)),
            "predict_wall_ms": elapsed,
            "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(cuda.max_memory_reserved(device)),
        }
        return depth, metrics

def summarize(records):
    if not records:
        return None
    summary = {"requests": len(records)}
    for key in ("forward_gpu_ms", "predict_wall_ms", "ros_round_trip_ms", "request_complete_ms"):
        values = np.array([r[key] for r in records], dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Invalid timing values: {key}")
        summary[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                        "p95": float(np.percentile(values, 95)),
                        "min": float(values.min()), "max": float(values.max())}
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        summary[key.replace("_bytes", "_gib")] = max(r[key] for r in records) / 2**30
    return summary
