import json
from pathlib import Path
import numpy as np
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def preview(rgb, truth, prediction, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')
    panels = [(rgb, title)]
    if truth is not None:
        valid = (truth > 0) & (truth <= 80)
        panels.append((np.ma.masked_where(~valid, truth), 'Ground truth; gray outside (0,80] m'))
    if prediction is not None:
        panels.append((prediction, 'Model prediction (m)'))
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 4 * len(panels)), layout='constrained', squeeze=False)
    image = None
    for index, (pixels, label) in enumerate(panels):
        ax = axes[index, 0]
        image = ax.imshow(pixels) if index == 0 else ax.imshow(pixels, cmap=cmap, vmin=0, vmax=80)
        ax.set_title(label)
        ax.set_axis_off()
    if len(panels) > 1:
        fig.colorbar(image, ax=list(axes[:, 0]), label='Camera-z depth (m)', shrink=0.85)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_first_result(rgb, depth, header, input_shape, directory, model_info):
    stamp = header.stamp
    prefix = directory / f'{stamp.sec}_{stamp.nanosec:09d}'
    np.save(f'{prefix}_depth.npy', depth)
    write_json(Path(f'{prefix}_info.json'), {**model_info, 'frame_id': header.frame_id, 'stamp': {'sec': stamp.sec, 'nanosec': stamp.nanosec}, 'units': 'meters', 'depth_convention': 'camera_z', 'network_input_shape': input_shape, 'output_shape': list(depth.shape)})
    path = Path(f'{prefix}_preview.png')
    preview(rgb, None, depth, model_info['model'], path)
    return path
