"""Shape-stratified KITTI speed workload and population-weighted summaries."""
import hashlib
import json
from pathlib import Path
import random
from collections import defaultdict
import numpy as np
from PIL import Image
from .paths import ROOT, KITTI_DIR
from .outputs import write_json

def inventory(processor=None):
    if processor is None:
        from transformers import AutoImageProcessor
        from .models import MODELS
        model,revision=MODELS['small']
        processor=AutoImageProcessor.from_pretrained(model,revision=revision,
                    cache_dir=str(ROOT/'checkpoints/huggingface'),use_fast=False)
    shapes={}
    groups=defaultdict(list)
    for path in sorted((KITTI_DIR/'training/image_2').glob('*.png')):
        if len(path.stem)!=6 or not path.stem.isascii() or not path.stem.isdigit():continue
        with Image.open(path) as image:w,h=image.size
        if (h,w) not in shapes:
            values=processor(images=np.zeros((h,w,3),dtype=np.uint8),return_tensors='np')['pixel_values']
            shapes[h,w]=tuple(int(n) for n in values.shape)
        groups[shapes[h,w]].append(int(path.stem))
    if not groups:raise ValueError('No KITTI RGB images found')
    return dict(groups)

def allocate(groups,count):
    total=sum(map(len,groups.values()))
    minimum={s:min(2,len(ids)) for s,ids in groups.items()}
    if not sum(minimum.values())<=count<=total:
        raise ValueError(f'Sample count must be in [{sum(minimum.values())},{total}]')
    result=dict(minimum)
    while sum(result.values())<count:
        eligible=[s for s in sorted(groups) if result[s]<len(groups[s])]
        choice=max(eligible,key=lambda s:count*len(groups[s])/total-result[s])
        result[choice]+=1
    return result

def create_plan(output,count=50,seed=249,processor=None):
    if output.exists():raise ValueError('Plan directory already exists')
    groups=inventory(processor)
    allocation=allocate(groups,count)
    total=sum(map(len,groups.values()))
    rng=random.Random(seed)
    strata=[]
    selected=[]
    for index,shape in enumerate(sorted(groups)):
        ids=sorted(rng.sample(groups[shape],allocation[shape]))
        selected.extend(ids)
        strata.append({'id':str(index),'input_shape':list(shape),
                       'population_images':len(groups[shape]),'sample_images':len(ids),
                       'population_weight':len(groups[shape])/total,'frames':ids})
    signature=[{'input_shape':list(shape),'frames':groups[shape]} for shape in sorted(groups)]
    plan={'schema_version':1,'dataset':'kitti','count':count,'seed':seed,
          'population_images':total,'frames':sorted(selected),'strata':strata,
          'population_sha256':hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),
          'scope':'KITTI RGB, batch-one preprocessed tensor shape distribution',
          'sampling':'proportional allocation with minimum 2 images per non-singleton stratum',
          'precision_note':'Sampling uncertainty is conditional on measured runs; not a guarantee of run-to-run thermal/driver variation'}
    output.mkdir(parents=True)
    write_json(output/'plan.json',plan)
    write_json(output/'frames.json',plan['frames'])
    print(f'Saved {count} distinct images from {total} images / {len(strata)} tensor shapes: {output / "plan.json"}')
    return plan

def validate_plan(plan):
    frames=plan['frames']
    if not frames or len(set(frames))!=len(frames):raise ValueError('Invalid plan frames')
    if any(type(f)is not int or not 0<=f<=999999 for f in frames):raise ValueError('Invalid frame number')
    if plan['count']!=len(frames):raise ValueError('Incorrect sample count')
    all_ids=[]
    for s in plan['strata']:
        all_ids.extend(s['frames'])
        if s['sample_images']!=len(s['frames']) or not 0<s['sample_images']<=s['population_images']:
            raise ValueError('Invalid stratum size')
        if not np.isclose(s['population_weight'],s['population_images']/plan['population_images']):raise ValueError('Invalid population weight')
    if sorted(all_ids)!=sorted(frames) or sum(s['population_images'] for s in plan['strata'])!=plan['population_images']:
        raise ValueError('Invalid stratum membership')

def weighted_summary(records,plan):
    validate_plan(plan)
    if {int(r['dataset_frame']) for r in records} != set(plan['frames']):
        raise ValueError('Timing records and sampling plan disagree')
    keys=('forward_gpu_ms','predict_wall_ms','ros_round_trip_ms','request_complete_ms')
    summary={}
    for key in keys:
        points=[]
        weights=[]
        estimate=0.
        conditional_bootstrap=np.zeros(1000)
        rng=np.random.default_rng(249)
        for stratum in plan['strata']:
            image_means=[]
            for number in stratum['frames']:
                samples=[r for r in records if int(r['dataset_frame'])==number]
                if not samples:raise ValueError('Missing sampled frame')
                if any(list(r['network_input_shape'])!=stratum['input_shape'] for r in samples):raise ValueError('Actual processed shape differs from sampling plan')
                values=np.array([r[key] for r in samples])
                if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid timing')
                image_means.append(float(values.mean()))
                points.extend(values.tolist())
                weights.extend([stratum['population_weight']/len(stratum['frames'])/len(values)]*len(values))
            image_means=np.array(image_means)
            estimate+=stratum['population_weight']*float(image_means.mean())
            sampled=rng.integers(0,len(image_means),size=(1000,len(image_means)))
            n,N=len(image_means),stratum['population_images']
            correction=np.sqrt((N-n)/(N-1)) if N>1 else 0.
            bootstrap=image_means.mean()+correction*(image_means[sampled].mean(axis=1)-image_means.mean())
            conditional_bootstrap+=stratum['population_weight']*bootstrap
        order=np.argsort(points)
        sorted_values=np.asarray(points)[order]
        cumulative=np.cumsum(np.asarray(weights)[order])
        def quantile(q):return float(sorted_values[min(np.searchsorted(cumulative,q),len(sorted_values)-1)])
        summary[key]={'population_weighted_mean':estimate,'weighted_median':quantile(.5),'weighted_p95':quantile(.95),
                      'image_sampling_mean_ci95':np.percentile(conditional_bootstrap,[2.5,97.5]).tolist()}
    summary['uncertainty_note']='Stratified image-cluster bootstrap with finite-population correction; repeats grouped by image. Conditional on this run; excludes between-run load/thermal variation.'
    return summary
