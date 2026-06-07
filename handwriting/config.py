from dataclasses import dataclass


@dataclass
class Config:
    # Model architecture
    lstm_size: int = 512
    output_mixtures: int = 30
    attention_mixtures: int = 10
    vocab_size: int = 65

    # Training
    batch_size: int = 32
    grad_accum_steps: int = 2       # effective batch = 64
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1000
    total_steps: int = 60_000       # ~8-10 h on M4 Air
    grad_clip: float = 10.0

    # Checkpointing / logging
    checkpoint_dir: str = "model/checkpoint"
    checkpoint_every: int = 500
    log_every: int = 50

    # Data
    data_dir: str = "model/data/processed"
    num_workers: int = 4
