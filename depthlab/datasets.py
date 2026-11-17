from pathlib import Path
import json
import random
import re
from .paths import ROOT
from .data import kitti, vkitti as vk
def select_kitti(dataset, count, seed, frame_list):
    if frame_list is not None:
        frames = json.loads(frame_list.read_text())
    else:
        images = dataset / 'training' / 'image_2'
        available = sorted((int(p.stem) for p in images.glob('*.png') if len(p.stem) == 6 and p.stem.isascii() and p.stem.isdigit()))
        if not 1 <= count <= len(available):
            raise ValueError(f'--count must be between 1 and {len(available)}')
        frames = sorted(random.Random(seed).sample(available, count))
    if not isinstance(frames, list) or not frames:
        raise ValueError('Frame list must be a nonempty JSON list of integers')
    if any((type(f) is not int or not 0 <= f <= 999999 for f in frames)):
        raise ValueError('Frame numbers must be integers between 0 and 999999')
    if len(set(frames)) != len(frames):
        raise ValueError('Frame list contains duplicates')
    for f in frames:
        for folder, suffix in [('image_2', '.png'), ('velodyne', '.bin'), ('calib', '.txt')]:
            path = dataset / 'training' / folder / f'{f:06d}{suffix}'
            if not path.is_file():
                raise ValueError(f'Missing dataset file: {path}')
    return frames
def select_vkitti(per_scene, seed, frame_list):
    if frame_list is not None:
        frames = json.loads(frame_list.read_text())
    else:
        if per_scene < 1:
            raise ValueError('--per-scene must be positive')
        rng = random.Random(seed)
        frames = []
        for scene in vk.SCENES:
            base = vk.DATASET / scene / 'clone/frames'
            rgb = {int(p.stem[4:]) for p in (base / 'rgb/Camera_0').glob('rgb_*.jpg') if re.fullmatch('rgb_[0-9]{5}', p.stem)}
            depth = {int(p.stem[6:]) for p in (base / 'depth/Camera_0').glob('depth_*.png') if re.fullmatch('depth_[0-9]{5}', p.stem)}
            if rgb != depth or len(rgb) < per_scene:
                raise ValueError(f'Missing or insufficient matched data in {scene}; finish prepare_vkitti2.py')
            frames.extend(({'scene': scene, 'frame': f} for f in sorted(rng.sample(sorted(rgb), per_scene))))
    if not isinstance(frames, list) or not frames:
        raise ValueError('Frame list must be a nonempty JSON list')
    keys = []
    for item in frames:
        if not isinstance(item, dict) or set(item) != {'scene', 'frame'}:
            raise ValueError('Each frame entry must contain only scene and frame')
        scene, frame = (item['scene'], item['frame'])
        if scene not in vk.SCENES or type(frame) is not int or (not 0 <= frame <= 99999):
            raise ValueError('Invalid scene or frame number')
        keys.append((scene, frame))
        for kind, suffix in [('rgb', '.jpg'), ('depth', '.png')]:
            path = vk.DATASET / scene / 'clone/frames' / kind / 'Camera_0' / f'{kind}_{frame:05d}{suffix}'
            if not path.is_file():
                raise ValueError(f'Missing input: {path}')
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate scene/frame entries')
    return frames
def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def load_sample(kind, key):
    if kind == 'kitti':
        frame = kitti.load_frame(key)
        return (frame.frame_id, frame.image, kitti.project_lidar(frame))
    if kind == 'vkitti2':
        rgb, truth = vk.read_pair(key['scene'], key['frame'])
        return (f"{key['scene']}_clone_Camera_0_{key['frame']:05d}", rgb, truth)
    raise ValueError('Dataset must be kitti or vkitti2')

def choose_frames(settings):
    selection = settings['selection']
    modes = [name for name in ('frames', 'frame_list', 'count', 'per_scene', 'all') if name in selection]
    if len(modes) != 1:
        raise ValueError('Choose exactly one selection: frames, frame_list, count, per_scene')
    frame_list = project_path(selection['frame_list']) if 'frame_list' in selection else None
    if 'all' in selection:
        if selection['all'] is not True:
            raise ValueError('all must be true')
        if settings['dataset'] == 'kitti':
            available = list((kitti.DATASET_DIR / 'training/image_2').glob('*.png'))
            return select_kitti(kitti.DATASET_DIR, len(available), 249, None)
        if settings['dataset'] == 'vkitti2':
            frames = []
            for scene in vk.SCENES:
                base = vk.DATASET / scene / 'clone/frames'
                rgb = {int(p.stem[4:]) for p in (base/'rgb/Camera_0').glob('rgb_*.jpg') if re.fullmatch(r'rgb_[0-9]{5}', p.stem)}
                depth = {int(p.stem[6:]) for p in (base/'depth/Camera_0').glob('depth_*.png') if re.fullmatch(r'depth_[0-9]{5}', p.stem)}
                if not rgb or rgb != depth:
                    raise ValueError(f'Missing or mismatched clone/Camera_0 data: {scene}')
                frames.extend({'scene': scene, 'frame': f} for f in sorted(rgb))
            return frames
        raise ValueError('Unknown dataset')
    if 'frames' in selection:
        frames = selection['frames']
        if not isinstance(frames, list) or not frames:
            raise ValueError('frames must be a nonempty list')
        identifiers = []
        for key in frames:
            if settings['dataset'] == 'kitti':
                if type(key) is not int or not 0 <= key <= 999999:
                    raise ValueError('Invalid KITTI frame')
                identifiers.append(key)
            else:
                if not isinstance(key, dict) or set(key) != {'scene', 'frame'} or key['scene'] not in vk.SCENES or (type(key['frame']) is not int) or (not 0 <= key['frame'] <= 99999):
                    raise ValueError('Invalid VKITTI frame')
                identifiers.append((key['scene'], key['frame']))
        if len(set(identifiers)) != len(frames):
            raise ValueError('Duplicate frames')
        return frames
    if settings['dataset'] == 'kitti':
        if 'per_scene' in selection:
            raise ValueError('per_scene is only valid for vkitti2')
        return select_kitti(kitti.DATASET_DIR, selection.get('count'), selection.get('seed', 249), frame_list)
    if settings['dataset'] == 'vkitti2':
        if 'count' in selection:
            raise ValueError('Use per_scene for vkitti2')
        return select_vkitti(selection.get('per_scene'), selection.get('seed', 249), frame_list)
    raise ValueError('Unknown dataset')

def frame_id(kind, key):
    return f'{key:06d}' if kind == 'kitti' else f"{key['scene']}_clone_Camera_0_{key['frame']:05d}"

def source_paths(kind, key):
    if kind == 'kitti':
        base = kitti.DATASET_DIR / 'training'
        return [base/folder/f'{key:06d}{suffix}' for folder,suffix in [('image_2','.png'),('velodyne','.bin'),('calib','.txt')]]
    base = vk.DATASET / key['scene'] / 'clone/frames'
    return [base/'rgb/Camera_0'/f"rgb_{key['frame']:05d}.jpg", base/'depth/Camera_0'/f"depth_{key['frame']:05d}.png"]

def source_signature(kind, key):
    return [{'path': str(p.resolve()), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in source_paths(kind,key)]

def load_rgb(kind, key):
    from PIL import Image
    import numpy as np
    with Image.open(source_paths(kind,key)[0]) as image:
        return np.array(image.convert('RGB'), dtype=np.uint8)
