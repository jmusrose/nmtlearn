import torch
import yaml
from pathlib import Path
from typing import Dict





def _check_path(path:str, allow_empty:bool=True) -> Path:
    """检查路径是否存在"""
    if path is not None:
        path = Path(path).absolute()
        if not allow_empty:
            assert path.exists(), f"{path}没有找到"
    return path


def load_config(cfg_files: str = "configs/default.yaml") -> Dict:
    """
    读取和解析Yaml文件
    :param cfg_files:YAML路径
    :return:config字典
    """
    cfg_file = _check_path(cfg_files, allow_empty=False)
    with cfg_file.open("r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if "model_dir" not in cfg:
        cfg["model_dir"] = cfg["training"]["model_dir"]
    return cfg
