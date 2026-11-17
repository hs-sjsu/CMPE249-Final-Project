import numpy as np
def evaluate(prediction, truth):
    if prediction.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError('Prediction and ground truth must have matching HxW shapes')
    for array in (prediction, truth):
        if array.dtype.kind not in 'fiu':
            raise ValueError('Depth arrays must contain real numbers')
        if not np.isfinite(array).all():
            raise ValueError('Depth arrays contain NaN or infinity')
    if np.any(truth < 0):
        raise ValueError('Ground truth must use zero for missing measurements')
    valid = (truth > 0) & (truth <= 80)
    if not valid.any():
        raise ValueError('No ground-truth measurements in (0, 80] meters')
    if np.any(prediction[valid] <= 0):
        raise ValueError('Nonpositive prediction at an evaluated pixel')
    masks = {'all': valid, '0-20 m': valid & (truth < 20), '20-40 m': valid & (truth >= 20) & (truth < 40), '40-80 m': valid & (truth >= 40)}
    scores = {}
    for label, mask in masks.items():
        count = int(mask.sum())
        if count == 0:
            scores[label] = {'pixels': 0}
            continue
        p = prediction[mask].astype(np.float64)
        g = truth[mask].astype(np.float64)
        error = p - g
        scores[label] = {'pixels': count, 'absrel': float(np.mean(np.abs(error) / g)), 'rmse_m': float(np.sqrt(np.mean(error ** 2))), 'delta1': float(np.mean(np.maximum(p / g, g / p) < 1.25)), 'median_prediction_to_truth': float(np.median(p / g)), 'mean_bias_m': float(np.mean(error))}
    return scores
def summarize(rows):
    summary = {}
    for label in ('all', '0-20 m', '20-40 m', '40-80 m'):
        scores = [r['scores'][label] for r in rows if r['scores'][label]['pixels']]
        summary[label] = {'images': len(scores), 'pixels': sum((s['pixels'] for s in scores)), 'image_mean': {key: float(np.mean([s[key] for s in scores])) if scores else None for key in ('absrel', 'rmse_m', 'delta1')}}
    return summary
