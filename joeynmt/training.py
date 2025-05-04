

from typing import Dict
from pathlib import Path
from joeynmt.helpers_for_ddp import get_logger, ddp_setup


def train(rank: int, world_size: int, cfg: Dict, skip_test: bool=False)
    """
    
    :param rank: 
    :param world_size: 
    :param cfg: 
    :param skip_test: 
    :return: 
    """
    if cfg.pop("use_ddp", False):
        # initialize ddp
        # TODO: make `master_addr` and `master_port` configurable
        ddp_setup(rank, world_size, master_addr="localhost", master_port=12355)

        # need to assign file handlers again, after multi-processes are spawned...
        get_logger(__name__, log_file=Path(cfg["model_dir"]) / "train.log")