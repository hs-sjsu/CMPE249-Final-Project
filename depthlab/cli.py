import argparse
import json
import os
from pathlib import Path
import numpy as np
def main():
    parser = argparse.ArgumentParser(prog='python3 -m depthlab')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('command', choices=('node', 'evaluate', 'benchmark', 'prepare', 'inspect', 'check', 'replay', 'verify'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    rest = args.arguments
    os.environ['DEPTHLAB_ROOT'] = str(args.root.resolve())
    try:
        if args.command == 'node':
            from .ros import node_main
            node_main(rest)
        elif args.command == 'benchmark':
            from .benchmark import main
            main(rest)
        elif args.command == 'evaluate':
            p = argparse.ArgumentParser()
            p.add_argument('--config', required=True)
            p.add_argument('--resume', action='store_true')
            option = p.parse_args(rest)
            from .experiment import run_experiment
            config_path=Path(option.config)
            if not config_path.is_absolute():config_path=args.root/config_path
            settings=json.loads(config_path.read_text())
            if option.resume:settings['resume']=True
            run_experiment(settings)
        elif args.command == 'verify':
            p = argparse.ArgumentParser()
            p.add_argument('--run', action='append', required=True)
            option = p.parse_args(rest)
            from .experiment import verify_run
            from .outputs import write_json
            checks = [verify_run(path) for path in option.run]
            write_json(args.root.resolve() / 'runs/migration/offline_checks.json', checks)
        elif args.command == 'check':
            from .checks import main
            main(rest)
        elif args.command == 'prepare':
            p = argparse.ArgumentParser()
            p.add_argument('dataset', choices=('kitti', 'vkitti2'))
            option = p.parse_args(rest)
            from importlib import import_module
            import_module(f'depthlab.downloads.{option.dataset}').prepare()
        elif args.command == 'inspect':
            p = argparse.ArgumentParser()
            p.add_argument('dataset', choices=('kitti', 'vkitti2'))
            p.add_argument('--frame', type=int, default=0)
            p.add_argument('--scene', default='Scene01')
            option = p.parse_args(rest)
            from .datasets import load_sample
            from .outputs import preview
            key = option.frame if option.dataset == 'kitti' else {'scene': option.scene, 'frame': option.frame}
            frame_id, rgb, truth = load_sample(option.dataset, key)
            folder = args.root.resolve() / 'runs/inspection'
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f'{frame_id}_ground_truth.npy', truth)
            preview(rgb, truth, None, frame_id, folder / f'{frame_id}.png')
            print(folder / f'{frame_id}.png')
        else:
            from .replay import main
            main(rest)
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError) as error:
        parser.exit(1, f'depthlab: {error}\n')
    except KeyboardInterrupt:
        parser.exit(1, 'Interrupted\n')
