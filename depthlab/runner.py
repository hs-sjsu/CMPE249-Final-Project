"""Streaming full/subset accuracy evaluation with shared inference profiling."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .datasets import choose_frames, load_sample, load_rgb, frame_id, source_signature, project_path
from .metrics import evaluate, summarize
from .measurement import PROFILE_TOPIC, PROTOCOL, summarize as summarize_performance
from .outputs import write_json, preview
from .storage import save_array, share_truth, array_hash, append_record, read_records, verify_artifacts
from .data import vkitti as vk

def run_experiment(settings):
    import rclpy
    from .models import MODELS
    from .ros import DepthClient
    kind, size = settings['dataset'], settings['model_size']
    frames = choose_frames(settings)
    ids = [frame_id(kind,key) for key in frames]
    profile = settings.get('collect_profile', True)
    warmup = settings.get('warmup_frames', 10)
    timeout = float(settings.get('timeout', 60))
    if type(settings.get('resume',False)) is not bool or type(profile) is not bool or type(warmup) is not int or warmup < 0 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('Invalid profiling/warmup/timeout settings')
    metadata_path = project_path(settings.get('model_info',f'runs/nodes/{size}/run_info.json'))
    model_bytes = metadata_path.read_bytes()
    model = json.loads(model_bytes)
    if (model.get('model'),model.get('revision')) != MODELS[size]:
        raise ValueError('Active node does not match selected checkpoint')
    if model.get('precision') != 'torch.float32' or model.get('sdpa_backend_policy') != 'math_only' or not model.get('node_session_id'):
        raise ValueError('Expected a recorded FP32/math session')
    if profile and not model.get('profiling_enabled'):
        raise ValueError('Restart the v0.2 node with profiling enabled (the default)')
    if not profile and model.get('profiling_enabled'):
        raise ValueError('For collect_profile:false, start the node with --no-profile')
    output = project_path(settings['output'])
    cache = project_path(settings.get('ground_truth_cache','runs/ground_truth_cache')) / kind
    source_files = ['runner.py','ros.py','measurement.py','models.py','metrics.py','datasets.py','data/kitti.py','data/vkitti.py','storage.py']
    hashes = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in source_files}
    identity = {k:v for k,v in model.items() if k!='node_session_id'}
    contract = {'settings':{k:v for k,v in settings.items() if k!='resume'},'frames':frames,'model':identity,'source_sha256':hashes}
    existing = output.exists()
    if existing:
        if not settings.get('resume',False):
            raise ValueError('Output exists; use resume:true or a new directory')
        if json.loads((output/'contract.json').read_text()) != contract:
            raise ValueError('Resume rejected: configuration/model/software/source changed')
    else:
        output.mkdir(parents=True)
        write_json(output/'contract.json',contract)
        write_json(output/'frames.json',frames)
        write_json(output/'experiment.json',settings)
    sessions = json.loads((output/'sessions.json').read_text()) if (output/'sessions.json').exists() else {}
    sessions[model['node_session_id']] = model
    write_json(output/'sessions.json',sessions)
    (output/'run_info.json').write_bytes(model_bytes)
    rows = read_records(output/'manifest.jsonl',repair_tail=settings.get('resume',False))
    if [r['dataset_frame'] for r in rows] != ids[:len(rows)]:
        raise ValueError('Progress journal does not match selected frame order')
    for key,row in zip(frames,rows):
        if source_signature(kind,key) != row['source_signature']:
            raise ValueError('Dataset files changed since the checkpoint')
        verify_artifacts(output,row)
    reference = project_path(settings['reference_run']) if settings.get('reference_run') else None
    ref_rows = {r['dataset_frame']:r for r in json.loads((reference/'manifest.json').read_text())} if reference else {}
    if len(rows)==len(frames) and (output/'baseline.json').exists():
        print(f'Already complete: {output / "baseline.json"}')
        return
    node = None
    rclpy.init()
    try:
        node = DepthClient()
        node.wait_ready(timeout,PROFILE_TOPIC if profile else None)
        warmup_shapes = set()
        # Warmup is bounded; it never performs an extra full-dataset pass.
        for i in range(warmup):
            rgb = load_rgb(kind,frames[i % len(frames)])
            _,_,telemetry = node.request(rgb,timeout,profile=profile)
            if telemetry:
                if telemetry['model'] != model:
                    raise ValueError('Node telemetry disagrees with its metadata file')
                warmup_shapes.add(tuple(telemetry['network_input_shape']))
        warm_sessions=json.loads((output/'warmup_sessions.json').read_text()) if (output/'warmup_sessions.json').exists() else {}
        warm_sessions[model['node_session_id']]={'requests':warmup,'input_shapes':[list(s) for s in sorted(warmup_shapes)]}
        write_json(output/'warmup_sessions.json',warm_sessions)
        seen_shapes=set(warmup_shapes)
        for key in frames[len(rows):]:
            if metadata_path.read_bytes()!=model_bytes:
                raise ValueError('Depth-node session changed; restart client with resume:true')
            started=time.perf_counter_ns()
            signature=source_signature(kind,key)
            identifier,rgb,truth=load_sample(kind,key)
            prep_ms=(time.perf_counter_ns()-started)/1e6
            started=time.perf_counter_ns()
            prediction,stamp,telemetry=node.request(rgb,timeout,profile=profile)
            request_ms=(time.perf_counter_ns()-started)/1e6
            if metadata_path.read_bytes()!=model_bytes or (telemetry and telemetry['model']!=model):
                raise ValueError('Active model/session changed')
            if source_signature(kind,key)!=signature:
                raise ValueError('Dataset source changed during the request')
            if telemetry:
                telemetry=dict(telemetry)
                telemetry.pop('model')
                shape=tuple(telemetry['network_input_shape'])
                telemetry['input_shape_previously_seen']=shape in seen_shapes
                seen_shapes.add(shape)
            started=time.perf_counter_ns()
            scores=evaluate(prediction,truth)
            metrics_ms=(time.perf_counter_ns()-started)/1e6
            rgb_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
            difference=None
            if reference:
                old=ref_rows[identifier]
                if old['rgb_sha256']!=rgb_hash:raise AssertionError('Reference RGB differs')
                np.testing.assert_array_equal(truth,np.load(reference/f'{identifier}_ground_truth.npy',allow_pickle=False))
                previous=np.load(reference/f'{identifier}_prediction.npy',allow_pickle=False)
                np.testing.assert_allclose(prediction,previous,rtol=1e-4,atol=1e-4)
                difference=float(np.max(np.abs(prediction-previous)))
            started=time.perf_counter_ns()
            save_array(output/f'{identifier}_prediction.npy',prediction)
            gt_hash,gt_cache=share_truth(cache,output,identifier,truth)
            if not rows:preview(rgb,truth,prediction,identifier,output/'first_preview.png')
            save_ms=(time.perf_counter_ns()-started)/1e6
            record={
                'dataset_frame':identifier,'rgb_sha256':rgb_hash,'ground_truth_sha256':gt_hash,
                'prediction_sha256':array_hash(prediction),'ground_truth_cache':gt_cache,
                'source_signature':signature,'node_session_id':model['node_session_id'],
                'stamp':{'sec':stamp[0],'nanosec':stamp[1]},'shape':list(truth.shape),'scores':scores,
                'coverage_percent':100*scores['all']['pixels']/truth.size,
                'reference_max_abs_difference_m':difference,
                'wall_ms':{'data_and_ground_truth':prep_ms,'ros_request':request_ms,'metrics':metrics_ms,'artifacts_and_preview':save_ms},
                'performance':telemetry,
            }
            if kind=='vkitti2':record.update({**key,'variation':'clone','camera':'Camera_0'})
            write_json(output/f'{identifier}_info.json',record)
            append_record(output/'manifest.jsonl',record)
            rows.append(record)
            print(f'{len(rows)}/{len(frames)} {identifier}: AbsRel={scores["all"]["absrel"]:.4f}',flush=True)
        measured=[r['performance'] for r in rows if r['performance'] is not None]
        report={
            'status':'complete','purpose':'VKITTI2 clone source-domain diagnostic' if kind=='vkitti2' else 'KITTI object-training evaluation pool',
            'model':model,'node_sessions':sessions,'selection':{'frames':frames},
            'protocol':'camera_z_meters; 0 < GT <= 80; no crop, clipping or alignment',
            'ground_truth_support':'dense rendered' if kind=='vkitti2' else 'projected sparse LiDAR',
            'training_overlap':'not verified','dense_sparse_coverage_difference':'not controlled',
            'aggregation':'equal image weight; exclude empty bins from that bin mean',
            'summary':summarize(rows),'performance_summary':summarize_performance(measured),
            'measurement_protocol':PROTOCOL if profile else None,'collect_profile':profile,'warmup_frames_per_client_session':warmup,
            'warmup_sessions':warm_sessions,
            'current_session_warmed_input_shapes':[list(s) for s in sorted(warmup_shapes)],
            'performance_scope':'instrumented streaming evaluation; loading, GT, scoring and saving outside inference timings; not the preloaded controlled benchmark' if profile else 'accuracy only; profiling disabled',
            'dataset_scope':'KITTI object-training image_2' if kind=='kitti' else 'five VKITTI2 scenes / clone / Camera_0 (prepared scope)',
            'mean_coverage_percent':float(np.mean([r['coverage_percent'] for r in rows])),
            'reference_run':str(reference) if reference else None,'regression_passed':bool(reference),
            'source_sha256':hashes,'metadata_persistence':'per-frame JSONL append/fsync; final JSON manifest once',
        }
        if kind=='vkitti2':report['summary_by_scene']={scene:summarize([r for r in rows if r['scene']==scene]) for scene in vk.SCENES if any(r['scene']==scene for r in rows)}
        write_json(output/'manifest.json',rows)
        write_json(output/'baseline.json',report)
        print(f'PASS: saved {output / "baseline.json"}')
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
