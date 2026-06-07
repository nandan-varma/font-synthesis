"""Load a trained checkpoint and generate stroke sequences."""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .alphabet import encode, MAX_CHAR_LEN
from .model import HandwritingRNN


def _auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> HandwritingRNN:
    device = device or _auto_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = HandwritingRNN(
        lstm_size=cfg.get('lstm_size', 512),
        output_mixtures=cfg.get('output_mixtures', 30),
        attention_mixtures=cfg.get('attention_mixtures', 10),
        vocab_size=cfg.get('vocab_size', 65),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def generate(
    lines: List[str],
    checkpoint_path: Optional[str] = None,
    biases: Optional[List[float]] = None,
    styles: Optional[List[int]] = None,
    style_dir: str = "model/style",
    device: Optional[torch.device] = None,
) -> List[np.ndarray]:
    """Generate stroke arrays for each line of text."""
    device = device or _auto_device()
    if checkpoint_path is None:
        checkpoint_path = _find_checkpoint()

    model = load_model(checkpoint_path, device)
    biases = biases or [0.5] * len(lines)

    results = []
    for i, (line, bias) in enumerate(zip(lines, biases)):
        encoded = encode(line)
        c = torch.from_numpy(encoded).long().unsqueeze(0)  # (1, T_c)
        c_len = len(encoded)
        style_strokes = None

        if styles is not None:
            idx = styles[i]
            sp = os.path.join(style_dir, f"style-{idx}-strokes.npy")
            cp = os.path.join(style_dir, f"style-{idx}-chars.npy")
            if os.path.exists(sp) and os.path.exists(cp):
                style_np = np.load(sp)
                style_chars = np.load(cp).tobytes().decode('utf-8', errors='ignore')
                combined = encode(style_chars + " " + line)
                c = torch.from_numpy(combined).long().unsqueeze(0)
                c_len = len(combined)
                style_strokes = torch.from_numpy(style_np).float().unsqueeze(0)

        strokes = model.sample(
            c=c,
            c_len=c_len,
            bias=bias,
            max_steps=40 * max(len(line), 1),
            style_strokes=style_strokes,
            device=device,
        )
        results.append(strokes)

    return results


def _find_checkpoint(checkpoint_dir: str = "model/checkpoint") -> str:
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest):
        info = torch.load(latest, map_location='cpu', weights_only=False)
        if os.path.exists(info['path']):
            return info['path']
    pts = sorted(Path(checkpoint_dir).glob("model-*.pt"),
                 key=lambda p: int(p.stem.split('-')[1]))
    if pts:
        return str(pts[-1])
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}. Train first: bash overnight.sh")
