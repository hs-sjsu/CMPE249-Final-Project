import json
from pathlib import Path
import re
import shutil
import zipfile
import zlib
import requests
from tqdm import tqdm
from ..paths import KITTI_DIR as DATASET_DIR
ARCHIVE_DIR = DATASET_DIR / "archives"
BASE_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti'
EXPECTED_IDS = {f'{i:06d}' for i in range(7481)}
FILES = [('data_object_calib.zip', 'calib', '.txt'), ('data_object_image_2.zip', 'image_2', '.png'), ('data_object_velodyne.zip', 'velodyne', '.bin')]
CHUNK_SIZE = 1024 * 1024

def download(session, filename):
    url = f'{BASE_URL}/{filename}'
    archive = ARCHIVE_DIR / filename
    partial = ARCHIVE_DIR / f'{filename}.part'
    metadata = ARCHIVE_DIR / f'{filename}.json'
    with session.head(url, allow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers['Content-Length'])
        etag = response.headers.get('ETag')
    source = {'url': url, 'bytes': total, 'etag': etag}
    try:
        previous = json.loads(metadata.read_text())
    except (FileNotFoundError, ValueError):
        previous = None
    if archive.is_file() and previous == source and (archive.stat().st_size == total) and zipfile.is_zipfile(archive):
        print(f'Using downloaded archive: {filename}')
        return archive
    if partial.exists() and (previous != source or not etag):
        partial.unlink()
    if partial.exists() and partial.stat().st_size > total:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    if shutil.disk_usage(DATASET_DIR).free < total - offset:
        raise OSError(f'Insufficient disk space to download {filename}')
    metadata.write_text(json.dumps(source, indent=2) + '\n')
    if offset < total:
        headers = {'Range': f'bytes={offset}-', 'If-Range': etag} if offset else {}
        with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if response.status_code == 206:
                expected = f'bytes {offset}-{total - 1}/{total}'
                if response.headers.get('Content-Range') != expected:
                    raise RuntimeError(f'Unexpected download range for {filename}')
            elif response.status_code == 200:
                offset = 0
            else:
                raise RuntimeError(f'Unexpected HTTP status: {response.status_code}')
            with partial.open('ab' if offset else 'wb') as output:
                with tqdm(total=total, initial=offset, unit='B', unit_scale=True, desc=filename) as progress:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            output.write(chunk)
                            progress.update(len(chunk))
    if partial.stat().st_size != total:
        raise RuntimeError(f'Incomplete download: {filename}. Rerun to resume.')
    if not zipfile.is_zipfile(partial):
        partial.unlink()
        raise RuntimeError(f'Downloaded file is not a valid ZIP: {filename}')
    partial.replace(archive)
    return archive

def matches_archive(path, member):
    """Check extracted files against ZIP size and CRC, including on reruns."""
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    crc = 0
    with path.open('rb') as source:
        while (chunk := source.read(CHUNK_SIZE)):
            crc = zlib.crc32(chunk, crc)
    return crc == member.CRC

def extract_training(archive, folder, suffix):
    pattern = f'training/{folder}/[0-9]{{6}}{re.escape(suffix)}'
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist() if re.fullmatch(pattern, m.filename)]
        ids = {Path(m.filename).stem for m in members}
        if ids != EXPECTED_IDS or len(members) != len(EXPECTED_IDS):
            raise RuntimeError(f'Unexpected training-frame list in {archive.name}')
        directory = DATASET_DIR / 'training' / folder
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.resolve().is_relative_to(DATASET_DIR.resolve()):
            raise RuntimeError('Extraction directory points outside dataset')
        needed = sum((m.file_size for m in members if not (DATASET_DIR / m.filename).is_file() or (DATASET_DIR / m.filename).stat().st_size != m.file_size))
        needed += max((m.file_size for m in members))
        if shutil.disk_usage(DATASET_DIR).free < needed:
            raise OSError(f'Insufficient disk space to extract {archive.name}')
        for member in tqdm(members, desc=f'Extracting {folder}', unit='file'):
            target = DATASET_DIR / member.filename
            if target.is_symlink():
                raise RuntimeError(f'Refusing to overwrite symlink: {target}')
            if matches_archive(target, member):
                continue
            temporary = target.with_suffix(target.suffix + '.part')
            if temporary.is_symlink():
                raise RuntimeError(f'Refusing to overwrite symlink: {temporary}')
            with source.open(member) as input_file, temporary.open('wb') as output:
                shutil.copyfileobj(input_file, output, CHUNK_SIZE)
            temporary.replace(target)

def prepare():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Dataset directory: {DATASET_DIR}')
    print('Approximately 41 GB of downloads; allow about 80 GB free for ZIPs + training files.')
    with requests.Session() as session:
        session.headers['Accept-Encoding'] = 'identity'
        for filename, folder, suffix in FILES:
            archive = download(session, filename)
            extract_training(archive, folder, suffix)
    frame_sets = [{p.stem for p in (DATASET_DIR / 'training' / folder).glob(f'*{suffix}')} for _, folder, suffix in FILES]
    if any((ids != EXPECTED_IDS for ids in frame_sets)):
        raise RuntimeError('Final check failed: RGB, LiDAR, and calibration IDs do not match')
    print(f'Ready: {len(EXPECTED_IDS)} matched RGB/LiDAR/calibration frames.')
