import hashlib
from pathlib import Path
import re
import shutil
import tarfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from ..paths import VKITTI_DIR as DATASET
from ..data.vkitti import SCENES
BASE_URL = 'https://download.europe.naverlabs.com/virtual_kitti_2.0.3'
CHECKSUMS = {'rgb': '1e00a143a397c2c53aa9720a868fe34a', 'depth': '1d34a96f870fc5d33b482df87844f5e4'}
BLOCK = 1024 * 1024

def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        while (chunk := stream.read(BLOCK)):
            digest.update(chunk)
    return digest.hexdigest()

def download(kind):
    folder = DATASET / 'archives'
    folder.mkdir(parents=True, exist_ok=True)
    filename = f'vkitti_2.0.3_{kind}.tar'
    archive = folder / filename
    partial = folder / f'{filename}.part'
    expected = CHECKSUMS[kind]
    if archive.exists():
        print(f'Checking checksum: {filename}', flush=True)
        if md5(archive) != expected:
            raise RuntimeError(f'Checksum mismatch: {archive}; move it aside and rerun')
        return archive
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(f'{BASE_URL}/{filename}', headers={'Range': f'bytes={offset}-'} if offset else {})
    try:
        response = urlopen(request, timeout=120)
    except HTTPError as error:
        if error.code == 416 and partial.exists() and (md5(partial) == expected):
            partial.replace(archive)
            return archive
        raise
    with response:
        if response.status == 206:
            match = re.fullmatch('bytes (\\d+)-(\\d+)/(\\d+)', response.headers.get('Content-Range', ''))
            if not match:
                raise RuntimeError('Missing or invalid Content-Range')
            start, end, total = map(int, match.groups())
            if start != offset or end != total - 1:
                raise RuntimeError('Unexpected resumed download range')
        elif response.status == 200:
            offset = 0
            length = response.headers.get('Content-Length')
            total = int(length) if length else None
        else:
            raise RuntimeError(f'Unexpected HTTP status: {response.status}')
        if total is not None and shutil.disk_usage(DATASET).free < total - offset:
            raise OSError('Not enough free space for the archive')
        print(f'Downloading {filename}; resuming at {offset:,} bytes', flush=True)
        downloaded = last_print = offset
        with partial.open('ab' if offset else 'wb') as output:
            while (chunk := response.read(BLOCK)):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_print >= 256 * BLOCK:
                    print(f'  {downloaded / 1000000000.0:.2f} GB received', flush=True)
                    last_print = downloaded
        if total is not None and downloaded != total:
            raise RuntimeError('Incomplete download; rerun to resume')
    print(f'Checking checksum: {filename}', flush=True)
    if md5(partial) != expected:
        raise RuntimeError(f'Checksum mismatch: {partial}; move it aside and rerun')
    partial.replace(archive)
    return archive

def extract_clone(archive, kind):
    extension = 'jpg' if kind == 'rgb' else 'png'
    pattern = re.compile(f'(Scene(?:01|02|06|18|20)/clone/frames/{kind}/Camera_0/{kind}_[0-9]{{5}}\\.{extension})$')
    keys = set()
    with tarfile.open(archive, 'r:') as source:
        selected = [(m, pattern.search(m.name)) for m in source]
        selected = [(m, match.group(1)) for m, match in selected if match]
        if not selected or any((not m.isfile() for m, _ in selected)):
            raise RuntimeError('Expected regular clone/Camera_0 files in archive')
        if shutil.disk_usage(DATASET).free < sum((m.size for m, _ in selected)):
            raise OSError('Not enough free space for selected extracted files')
        for member, relative in selected:
            target = DATASET / relative
            temporary = target.with_suffix(target.suffix + '.part')
            if not target.resolve().is_relative_to(DATASET.resolve()) or target.is_symlink() or temporary.is_symlink():
                raise RuntimeError('Unexpected extraction path or symlink')
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, temporary.open('wb') as output:
                shutil.copyfileobj(incoming, output, BLOCK)
            if temporary.stat().st_size != member.size:
                raise RuntimeError('Extracted file has an incorrect size')
            temporary.replace(target)
            key = (Path(relative).parts[0], int(target.stem.split('_')[1]))
            if key in keys:
                raise RuntimeError('Duplicate frame in archive')
            keys.add(key)
    print(f'Extracted {len(keys):,} {kind} frames', flush=True)
    return keys

def prepare():
    rgb = extract_clone(download("rgb"), "rgb")
    depth = extract_clone(download("depth"), "depth")
    if rgb != depth or not all(any(s == scene for s, _ in rgb) for scene in SCENES):
        raise RuntimeError("Missing or mismatched VKITTI pairs")
    print(f"Ready: {len(rgb)} matched clone/Camera_0 frames")
