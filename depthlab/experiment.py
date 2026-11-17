import hashlib
import json
from pathlib import Path
import numpy as np
from .paths import ROOT
from .datasets import project_path, load_sample, choose_frames
from .data import vkitti as vk
from .metrics import evaluate, summarize
from .outputs import write_json, preview
def assert_scores(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError('Different score bins')
    for label in actual:
        if set(actual[label]) != set(expected[label]):
            raise AssertionError(f'Different score fields: {label}')
        for name, value in actual[label].items():
            if not np.isclose(value, expected[label][name], rtol=1e-12, atol=1e-12):
                raise AssertionError(f'Score changed: {label}/{name}')

def verify_run(directory):
    directory = project_path(directory)
    report = json.loads((directory / 'baseline.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Reference run is incomplete')
    rows = json.loads((directory / 'manifest.json').read_text())
    kind = 'vkitti2' if 'VKITTI' in report['purpose'] else 'kitti'
    for row in rows:
        key = {'scene': row['scene'], 'frame': row['frame']} if kind == 'vkitti2' else int(row['dataset_frame'])
        frame_id, rgb, truth = load_sample(kind, key)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row['rgb_sha256']:
            raise AssertionError(f'RGB changed: {frame_id}')
        previous_truth = np.load(directory / f'{frame_id}_ground_truth.npy', allow_pickle=False)
        if not np.array_equal(truth, previous_truth):
            raise AssertionError(f'Ground truth changed: {frame_id}')
        prediction = np.load(directory / f'{frame_id}_prediction.npy', allow_pickle=False)
        assert_scores(evaluate(prediction, truth), row['scores'])
    for label, summary in summarize(rows).items():
        old = report['summary'][label]
        if (summary['images'], summary['pixels']) != (old['images'], old['pixels']):
            raise AssertionError('Summary counts changed')
        for metric, value in summary['image_mean'].items():
            previous = old['image_mean'][metric]
            if value is None and previous is None:
                continue
            if not np.isclose(value, previous, rtol=1e-12, atol=1e-12):
                raise AssertionError('Summary changed')
    print(f'PASS: {len(rows)} saved RGB/GT/metric checks: {directory}')
    return {'run': str(directory), 'frames': len(rows), 'passed': True}

def run_experiment(settings):
    from .runner import run_experiment as run
    return run(settings)
