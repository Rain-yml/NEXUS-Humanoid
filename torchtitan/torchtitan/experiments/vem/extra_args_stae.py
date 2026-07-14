from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict, Any


@dataclass
class Training:
    num_workers: int = 0
    drop_last: bool = True
    pin_memory: bool = True    
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class Validation:
    enabled: bool = False
    dataset: str = ""
    interval: int = 1000
    num_workers: int = 0
    drop_last: bool = False
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    pin_memory: bool = True
    batch_size: int = 1

@dataclass
class EMA:
    enabled: bool = False
    beta: float = 0.9999
    update_after_step: int = 100
    update_every: int = 10
    inv_gamma: float = 1.0
    power: float = 1.0
    min_value: float = 0.0

@dataclass
class JobConfig:
    training: Training = field(default_factory=Training)
    validation: Validation = field(default_factory=Validation)

    ema: EMA = field(default_factory=EMA)