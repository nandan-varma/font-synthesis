#!/usr/bin/env python3
"""
Prepare IAM On-Line Handwriting data for training.

Usage:
    python scripts/prepare_data.py              # extract archives + process data
    python scripts/prepare_data.py --extract-only  # extract archives only

Required archives in project root:
    xml.tgz.zip          (available - original form XMLs for writer IDs)
    lineStrokes-all.tar.gz   (download from IAM On-Line database)
    ascii-all.tar.gz         (download from IAM On-Line database)

Register at: https://fki.tic.teia.ch/databases/iam-on-line-handwriting-database
"""

import argparse
import os
import tarfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.signal import savgol_filter

PROCESSED_DIR = "model/data/processed"
RAW_DIR = "model/data/raw"
MAX_STROKE_LEN = 1200
MAX_CHAR_LEN = 75

_alphabet = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z',
]
_alpha_to_num = {c: i for i, c in enumerate(_alphabet)}


# ── Archive extraction ──────────────────────────────────────────────────────

def _extract(archive: str, dest: str, is_zip_wrapping_tgz: bool = False):
    if not os.path.exists(archive):
        return False
    dest_path = Path(dest)
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"  Already extracted: {archive}")
        return True
    print(f"  Extracting {archive} → {dest}/")
    dest_path.mkdir(parents=True, exist_ok=True)
    if is_zip_wrapping_tgz:
        with zipfile.ZipFile(archive, 'r') as z:
            inner = z.namelist()[0]
            with z.open(inner) as f:
                with tarfile.open(fileobj=f, mode='r:gz') as t:
                    t.extractall(dest)
    else:
        with tarfile.open(archive, 'r:gz') as t:
            t.extractall(dest)
    print(f"    Done.")
    return True


def extract_archives():
    print("Extracting archives...")
    _extract("xml.tgz.zip",            os.path.join(RAW_DIR, "original"),    is_zip_wrapping_tgz=True)
    _extract("lineStrokes-all.tar.gz", os.path.join(RAW_DIR, "lineStrokes"), is_zip_wrapping_tgz=False)
    _extract("ascii-all.tar.gz",       os.path.join(RAW_DIR, "ascii"),       is_zip_wrapping_tgz=False)


# ── Stroke processing ───────────────────────────────────────────────────────

def _align(coords: np.ndarray) -> np.ndarray:
    coords = np.copy(coords)
    x_mat = np.c_[np.ones(len(coords)), coords[:, 0]]
    coef  = np.linalg.lstsq(x_mat, coords[:, 1], rcond=None)[0]
    theta = float(np.arctan(coef[1]))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    coords[:, :2] = coords[:, :2] @ R - coef[0]
    return coords


def _denoise(coords: np.ndarray) -> np.ndarray:
    strokes = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    out = []
    for s in strokes:
        if len(s) == 0:
            continue
        if len(s) >= 7:
            s = s.copy()
            s[:, 0] = savgol_filter(s[:, 0], 7, 3, mode='nearest')
            s[:, 1] = savgol_filter(s[:, 1], 7, 3, mode='nearest')
        out.append(s)
    return np.vstack(out) if out else coords


def get_stroke_sequence(filename: str) -> np.ndarray | None:
    try:
        root = ElementTree.parse(filename).getroot()
    except Exception:
        return None
    stroke_sets = [e for e in root if e.tag == 'StrokeSet']
    if not stroke_sets:
        return None
    coords = []
    for stroke in stroke_sets[0]:
        for i, pt in enumerate(stroke):
            coords.append([int(pt.attrib['x']), -int(pt.attrib['y']),
                           int(i == len(stroke) - 1)])
    if not coords:
        return None
    coords = np.array(coords, dtype=np.float32)
    coords = _align(coords)
    coords = _denoise(coords)
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    offsets = np.concatenate([np.array([[0, 0, 1]], dtype=np.float32), offsets], axis=0)
    offsets = offsets[:MAX_STROKE_LEN]
    norms = np.linalg.norm(offsets[:, :2], axis=1)
    med = float(np.median(norms[norms > 0])) or 1.0
    offsets[:, :2] /= med
    return offsets


def _encode(s: str) -> np.ndarray:
    return np.array([_alpha_to_num.get(c, 0) for c in s] + [0], dtype=np.int8)


def _parse_ascii(filename: str):
    text = open(filename, 'r', errors='replace').read()
    text = text.replace(r'%%%%%%%%%%%', '\n')
    parts = [p.strip() for p in text.split('\n')]
    try:
        idx = parts.index('CSR:')
        lines = [l.strip() for l in parts[idx + 2:] if l.strip()]
    except ValueError:
        lines = []
    return [_encode(l)[:MAX_CHAR_LEN] for l in lines]


def _collect():
    ascii_root = os.path.join(RAW_DIR, "ascii")
    ls_root    = os.path.join(RAW_DIR, "lineStrokes")
    orig_root  = os.path.join(RAW_DIR, "original")
    blacklist  = set()
    bl_path = "model/data/blacklist.npy"
    if os.path.exists(bl_path):
        blacklist = set(np.load(bl_path, allow_pickle=True))

    stroke_fnames, transcriptions, writer_ids = [], [], []
    for dirpath, dirnames, filenames in os.walk(ascii_root):
        if dirnames:
            continue
        for fname in sorted(filenames):
            if fname.startswith('.') or not fname.endswith('.txt'):
                continue
            full = os.path.join(dirpath, fname)
            if full.endswith('z01-000z.txt'):
                continue

            stem = os.path.splitext(fname)[0]
            last = stem[-1] if stem[-1].isalpha() else ''
            ls_dir = dirpath.replace('ascii', 'lineStrokes')
            prefix = os.path.split(dirpath)[-1] + last + '-'
            if not os.path.isdir(ls_dir):
                continue

            ls_files = sorted(f for f in os.listdir(ls_dir) if f.startswith(prefix))
            if not ls_files:
                continue

            writer_id = 0
            orig_dir  = dirpath.replace('ascii', 'original')
            orig_xml  = os.path.join(orig_dir, 'strokes' + last + '.xml')
            if os.path.exists(orig_xml):
                try:
                    root = ElementTree.parse(orig_xml).getroot()
                    gen  = root.find('General')
                    if gen is not None and len(gen):
                        writer_id = int(gen[0].attrib.get('writerID', '0'))
                except Exception:
                    pass

            seqs = _parse_ascii(full)
            if len(seqs) != len(ls_files):
                continue

            for seq, lsf in zip(seqs, ls_files):
                if lsf in blacklist:
                    continue
                stroke_fnames.append(os.path.join(ls_dir, lsf))
                transcriptions.append(seq)
                writer_ids.append(writer_id)

    return stroke_fnames, transcriptions, writer_ids


# ── Main prepare ────────────────────────────────────────────────────────────

def prepare():
    print("Collecting sample list...")
    fnames, transcriptions, _ = _collect()
    n = len(fnames)
    print(f"Found {n} samples — processing...")

    x     = np.zeros([n, MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([n], dtype=np.int16)
    c     = np.zeros([n, MAX_CHAR_LEN],    dtype=np.int8)
    c_len = np.zeros([n], dtype=np.int8)
    valid = np.zeros([n], dtype=bool)

    for i, (sfname, ci) in enumerate(zip(fnames, transcriptions)):
        if i % 500 == 0:
            print(f"  {i}/{n}")
        strokes = get_stroke_sequence(sfname)
        if strokes is None:
            continue
        if np.any(np.linalg.norm(strokes[:, :2], axis=1) > 60):
            continue
        valid[i] = True
        x[i, :len(strokes)] = strokes
        x_len[i] = len(strokes)
        cl = min(len(ci), MAX_CHAR_LEN)
        c[i, :cl] = ci[:cl]
        c_len[i] = cl

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(f"{PROCESSED_DIR}/x.npy",     x[valid])
    np.save(f"{PROCESSED_DIR}/x_len.npy", x_len[valid])
    np.save(f"{PROCESSED_DIR}/c.npy",     c[valid])
    np.save(f"{PROCESSED_DIR}/c_len.npy", c_len[valid])
    print(f"Saved {valid.sum()} valid samples → {PROCESSED_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-only", action="store_true",
                    help="Only extract archives, skip numpy processing")
    args = ap.parse_args()

    extract_archives()
    if not args.extract_only:
        ascii_dir = os.path.join(RAW_DIR, "ascii")
        ls_dir    = os.path.join(RAW_DIR, "lineStrokes")
        if not os.path.isdir(ascii_dir) or not os.path.isdir(ls_dir):
            print("\nERROR: Missing IAM On-Line stroke data.")
            print("Register at: https://fki.tic.teia.ch/databases/iam-on-line-handwriting-database")
            print("Download and place in project root:")
            print("  lineStrokes-all.tar.gz")
            print("  ascii-all.tar.gz")
            raise SystemExit(1)
        prepare()
