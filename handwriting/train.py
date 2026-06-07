"""Training loop — run via: python -m handwriting.train  or  bash overnight.sh"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .dataset import get_dataloader
from .loss import mdn_loss
from .model import HandwritingRNN


def _auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_latest(checkpoint_dir: str):
    p = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(p):
        info = torch.load(p, map_location='cpu', weights_only=False)
        if os.path.exists(info['path']):
            return info['path']
    return None


def _save(model, optimizer, step: int, cfg: Config):
    path = os.path.join(cfg.checkpoint_dir, f"model-{step}.pt")
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': vars(cfg),
    }, path)
    torch.save({'step': step, 'path': path},
               os.path.join(cfg.checkpoint_dir, "latest.pt"))
    print(f"  checkpoint → {path}")


def train(cfg: Config | None = None):
    if cfg is None:
        cfg = Config()

    device = _auto_device()
    if device.type == "mps":
        torch.backends.mps.enable_fallback_computations = True
    print(f"Device : {device}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_loader = get_dataloader(cfg, "train")
    val_loader   = get_dataloader(cfg, "val")
    train_iter   = iter(train_loader)

    model = HandwritingRNN(
        lstm_size=cfg.lstm_size,
        output_mixtures=cfg.output_mixtures,
        attention_mixtures=cfg.attention_mixtures,
        vocab_size=cfg.vocab_size,
    ).to(device)

    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=cfg.total_steps // cfg.grad_accum_steps,
        pct_start=min(cfg.lr_warmup_steps / cfg.total_steps, 0.3),
        anneal_strategy='cos',
    )

    start_step = 0
    latest = _find_latest(cfg.checkpoint_dir)
    if latest:
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        print(f"Resumed from step {start_step}")

    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for step in range(start_step, cfg.total_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x     = batch['x'].to(device)                          # (B, T, 3)
        c     = batch['c'].to(device)                          # (B, T_c)
        x_len = torch.tensor(batch['x_len'], device=device)
        c_len = torch.tensor(batch['c_len'], device=device)

        x_in   = x[:, :-1, :]        # (B, T-1, 3) inputs
        y      = x[:, 1:,  :]        # (B, T-1, 3) targets
        seq_len = (x_len - 1).clamp(min=1)

        params = model(x_in, c, c_len, seq_len)
        loss   = mdn_loss(params, y, seq_len) / cfg.grad_accum_steps
        loss.backward()
        running_loss += loss.item() * cfg.grad_accum_steps

        if (step + 1) % cfg.grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % cfg.log_every == 0:
            avg  = running_loss / cfg.log_every
            lr   = scheduler.get_last_lr()[0]
            sps  = cfg.log_every / (time.time() - t0)
            eta  = (cfg.total_steps - step - 1) / sps / 3600
            print(f"step {step+1:>6} | loss {avg:.4f} | lr {lr:.2e} | {sps:.1f} step/s | ETA {eta:.1f}h")
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % cfg.checkpoint_every == 0:
            _save(model, optimizer, step + 1, cfg)

    _save(model, optimizer, cfg.total_steps, cfg)
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lstm-size",           type=int,   default=512)
    p.add_argument("--output-mixtures",     type=int,   default=30)
    p.add_argument("--attention-mixtures",  type=int,   default=10)
    p.add_argument("--batch-size",          type=int,   default=32)
    p.add_argument("--grad-accum",          type=int,   default=2)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--steps",               type=int,   default=60_000)
    p.add_argument("--checkpoint-every",    type=int,   default=500)
    p.add_argument("--log-every",           type=int,   default=50)
    p.add_argument("--output-dir",          type=str,   default="model/checkpoint")
    p.add_argument("--data-dir",            type=str,   default="model/data/processed")
    args = p.parse_args()

    cfg = Config(
        lstm_size=args.lstm_size,
        output_mixtures=args.output_mixtures,
        attention_mixtures=args.attention_mixtures,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        total_steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        checkpoint_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    train(cfg)
