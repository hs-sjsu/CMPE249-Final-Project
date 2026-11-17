"""Atomic arrays, immutable shared ground truth, and an append-only progress log."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes(order="C")).hexdigest()

def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def share_truth(cache, output, frame_id, truth):
    digest = array_hash(truth)
    source = cache / f"{frame_id}_{digest}.npy"
    if source.exists():
        if not np.array_equal(np.load(source, allow_pickle=False), truth):
            raise ValueError("Ground-truth cache entry is corrupt")
    else:
        save_array(source, truth)
    target = output / f"{frame_id}_ground_truth.npy"
    if target.exists():
        if not np.array_equal(np.load(target, allow_pickle=False), truth):
            raise ValueError("Existing run ground truth differs")
    else:
        try:
            os.link(source, target)
        except OSError:
            # A copy remains portable when hard links are unavailable.
            temporary = target.with_suffix(".npy.tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(target)
    return digest, str(source.resolve())

def append_record(path, record):
    payload = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

def read_records(path, repair_tail=False):
    if not path.exists():
        return []
    result = []
    with path.open("rb+") as stream:
        good_end = 0
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("Uncommitted journal line")
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if repair_tail and not stream.read(1):
                    stream.truncate(good_end)
                    break
                raise ValueError("Corrupt progress journal")
            result.append(record)
            good_end = stream.tell()
    return result

def verify_artifacts(output, record):
    frame = record["dataset_frame"]
    for suffix, key in [("prediction", "prediction_sha256"), ("ground_truth", "ground_truth_sha256")]:
        value = np.load(output / f"{frame}_{suffix}.npy", allow_pickle=False)
        if list(value.shape) != record["shape"] or array_hash(value) != record[key]:
            raise ValueError(f"Saved artifact changed: {frame}/{suffix}")
