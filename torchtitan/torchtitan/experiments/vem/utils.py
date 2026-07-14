from typing import Optional
from dataclasses import asdict
import logging
import os
import toml


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

logger = logging.getLogger()


def init_logger(level: int = logging.INFO, log_file: Optional[str] = None, save_log_file_rank_zero_only: bool = True):
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        if save_log_file_rank_zero_only and get_rank() != 0:
            pass
        else:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            formatter = logging.Formatter(
                "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


def dump_config(config, dump_path: str):
    if get_rank() == 0:
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "w") as f:
            toml.dump(asdict(config), f)
