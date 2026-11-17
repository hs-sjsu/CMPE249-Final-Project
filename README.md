# Depthlab: Monocular Depth Evaluation with ROS 2

Depthlab evaluates **Depth Anything V2 Metric Outdoor** on real KITTI images and benchmarks inference on AMD Radeon GPUs. It combines dataset preparation, LiDAR projection, ROS 2 inference, accuracy evaluation, and performance measurement.

> Documentation is in progress. This README will be expanded with detailed reproduction instructions, figures, and results.

## Overview

- Compare the Small and Large outdoor metric-depth checkpoints.
- Evaluate RGB-only predictions against projected KITTI Velodyne depth.
- Use Virtual KITTI 2 for synthetic-domain diagnostics.
- Benchmark batch-size-one inference on the RX 9070 XT and RX 6750 XT.

The project evaluates released checkpoints; it does not train a new model.

## Requirements

The development environment uses **Nix flakes and direnv**. GPU execution requires a working **AMD ROCm** setup. The environment definition and pinned inputs are in `flake.nix` and `flake.lock`.

Run all commands from the repository root.

## Quick start

Load the environment and check the basic evaluation logic:

```bash
direnv allow
python3 -m depthlab check unit
```

Download and prepare datasets when setting up a fresh installation:

```bash
python3 -m depthlab prepare kitti
python3 -m depthlab prepare vkitti2
```

Start the Small inference node in one terminal:

```bash
python3 -m depthlab node --model-size small
```

For the tested **RX 6750 XT** setup, use this compatibility override when starting the node:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  python3 -m depthlab node --model-size small
```

Once the node reports ready, run the evaluator in another terminal:

```bash
python3 -m depthlab evaluate \
  --config experiments/depthlab/v2/kitti_small_full_rx9070xt.json
```

This example is the full RX 9070 XT accuracy evaluation. The evaluator publishes its own images. For Large, restart the node with `--model-size large` and use `kitti_large_full_rx9070xt.json`.

## Speed benchmarking

Create a fixed sampling plan once:

```bash
python3 -m depthlab benchmark plan \
  --count 50 --seed 249 \
  --output runs/benchmark_plan/representative50
```

With the matching node running:

```bash
python3 -m depthlab benchmark client \
  --plan runs/benchmark_plan/representative50/plan.json \
  --output runs/performance/rx9070xt/small_representative50.json
```

Reuse the same plan across GPUs and checkpoints, with separate output filenames. Each default run uses 50 warmup requests and 150 measured requests. Images are preloaded into CPU RAM; the model processes **one image per request**.

## Evaluation and ROS interfaces

The real-data evaluation uses the **7,481-image KITTI object-training pool**. Predictions are camera-z depth in meters. Only ground-truth pixels satisfying **0 < depth <= 80 m** contribute to accuracy metrics.

Metrics include AbsRel, RMSE, and delta < 1.25, overall and in 0–20 m, 20–40 m, and 40–80 m distance bins. Results use per-image averaging without an evaluation crop or scale alignment. Upstream training overlap has not been independently audited.

| Topic | Content |
| --- | --- |
| `/kitti/camera_left/image_rect` | Input RGB image (`rgb8`) |
| `/depth/metric` | Predicted depth (`32FC1`, meters) |
| `/depth/profile` | JSON timing, memory, and model metadata |

## Project files

- `depthlab/`: implementation and built-in checks.
- `experiments/depthlab/v2/`: experiment configurations.
- `docs/`: project documentation.
- `results/`: selected results and figures for publication.

Datasets, downloaded weights, and generated runs live in `dataset/`, `checkpoints/`, and `runs/`. These directories are excluded from Git. Virtual KITTI preparation extracts the five-scene `clone/Camera_0` subset. Model files are downloaded on first use.
