"""Preloaded subset speed mode using the same ROS node and measurements as evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import rclpy
from .paths import ROOT, KITTI_DIR
from .ros import DepthNode, DepthClient
from .measurement import PROFILE_TOPIC, PROTOCOL, summarize
from .datasets import load_rgb
from .outputs import write_json
from .sampling import create_plan, validate_plan, weighted_summary

class ProfileNode(DepthNode):
    def __init__(self,size):
        super().__init__(ROOT/'runs/nodes'/size,size,profile=True,save_first=False)

ProfileClient = DepthClient

def run_client(node,args):
    plan=json.loads(args.plan.read_text()) if getattr(args,'plan',None) else None
    if plan:
        validate_plan(plan)
        frames=plan['frames']
        if Path(args.dataset).resolve()!=KITTI_DIR.resolve():raise ValueError('Plan requires its original KITTI root')
    else:
        frames=json.loads(args.frame_list.read_text())
    if not isinstance(frames,list) or not frames or any(type(f)is not int or not 0<=f<=999999 for f in frames) or len(set(frames))!=len(frames):
        raise ValueError('Invalid frame list')
    if args.output.exists():raise ValueError('Output exists; use a fresh filename')
    # RGB only. No LiDAR or rendered ground truth is read by speed mode.
    if Path(args.dataset).resolve()!=KITTI_DIR.resolve():
        from PIL import Image
        images=[]
        for f in frames:
            with Image.open(args.dataset/'training/image_2'/f'{f:06d}.png') as im:
                images.append(np.array(im.convert('RGB'),dtype=np.uint8))
    else:images=[load_rgb('kitti',f) for f in frames]
    node.wait_ready(args.timeout,PROFILE_TOPIC)
    model=None
    records=[]
    try:
        for pass_number in range(args.warmup_passes+args.repeats):
            warming=pass_number<args.warmup_passes
            for number,rgb in zip(frames,images):
                record=node.measure(rgb,args.timeout)
                if plan:
                    expected=next(s['input_shape'] for s in plan['strata'] if number in s['frames'])
                    if list(record['network_input_shape'])!=expected:raise ValueError('Processed input shape differs from plan')
                current=record.pop('model')
                if model is None:model=current
                if current!=model:raise ValueError('Node configuration/session changed')
                if not warming:
                    records.append({**record,'dataset_frame':f'{number:06d}','repeat':pass_number-args.warmup_passes+1})
            print(f'{"Warmup" if warming else "Measured"} pass complete',flush=True)
        report={
            'status':'complete','model':model,'frames':frames,
            'rgb_sha256':[hashlib.sha256(rgb.tobytes()).hexdigest() for rgb in images],
            'warmup_passes':args.warmup_passes,'measured_passes':args.repeats,'measured_requests':len(records),
            'protocol':{**PROTOCOL,'workload':'preloaded KITTI RGB; no GT, scoring, previews or per-request disk writes'},
            'summary':summarize(records),'sampling_plan':plan,
            'population_weighted_summary':weighted_summary(records,plan) if plan else None,
            'records':records,
            'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['benchmark.py','measurement.py','ros.py','sampling.py']},
        }
        write_json(args.output,report)
        print(json.dumps(report['population_weighted_summary'] or report['summary'],indent=2))
        print(f'Saved: {args.output.resolve()}')
    except (Exception,KeyboardInterrupt):
        if records:write_json(args.output.with_suffix('.partial.json'),{'status':'incomplete','model':model,'records':records})
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='role',required=True)
    server=commands.add_parser('node')
    server.add_argument('--model-size',choices=('small','large'),default='small')
    plan=commands.add_parser('plan')
    plan.add_argument('--count',type=int,default=50)
    plan.add_argument('--seed',type=int,default=249)
    plan.add_argument('--output',type=Path,default=ROOT/'runs/benchmark_plan/representative50')
    client=commands.add_parser('client')
    source=client.add_mutually_exclusive_group(required=True)
    source.add_argument('--plan',type=Path)
    source.add_argument('--frame-list',type=Path)
    client.add_argument('--dataset',type=Path,default=KITTI_DIR)
    client.add_argument('--output',type=Path,required=True)
    client.add_argument('--warmup-passes',type=int,default=1)
    client.add_argument('--repeats',type=int,default=3)
    client.add_argument('--timeout',type=float,default=60)
    args,ros_args=parser.parse_known_args(argv)
    if args.role=='plan':
        create_plan(args.output,args.count,args.seed)
        return
    if args.role=='client' and (args.warmup_passes<1 or args.repeats<1 or not np.isfinite(args.timeout) or args.timeout<=0):
        parser.error('Warmup, repeats and timeout must be positive')
    node=None
    rclpy.init(args=[sys.argv[0],*ros_args])
    try:
        if args.role=='node':
            node=ProfileNode(args.model_size)
            while rclpy.ok() and node.failure is None:rclpy.spin_once(node,timeout_sec=.5)
            if node.failure:raise RuntimeError(node.failure)
        else:
            node=ProfileClient()
            run_client(node,args)
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
