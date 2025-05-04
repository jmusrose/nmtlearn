import argparse
import shutil

import torch.cuda
import torch.multiprocessing as mp

from joeynmt.config import load_config, _check_path
from pathlib import Path
from joeynmt.helpers import make_model_dir
from joeynmt.helpers_for_ddp import get_logger
from joeynmt.training import train

def main():
    ap = argparse.ArgumentParser("joeynmt")
    ap.add_argument("mode", choices=["train", "test", "translate"], help="选择训练模式")
    ap.add_argument("config_path", metavar="config-path", type=str, help="配置文件路径")
    ap.add_argument("-o", "--output-path", type=str, help="翻译所保存的路径")
    ap.add_argument("-a", "--save-attention", action="store_true", help="注意力机制可视化")
    ap.add_argument("-s", "--save-scores", action="store_true", help="保存分数")
    ap.add_argument("-t", "--skip-test", action="store_true", help="训练之后跳过测试阶段")
    ap.add_argument("-d", "--use-ddp", action="store_true", help="Invoke DDP environment")

    args = ap.parse_args()

    #读配置文件
    cfg = load_config(Path(args.config_path))
    # print(cfg)

    #做模型地址
    if args.mode == "train":
        make_model_dir(
            Path(cfg["model_dir"]), overwrite=cfg["training"].get("overwrite", False)
        )
    model_dir = _check_path(cfg["model_dir"], allow_empty=False)
    if args.mode == "train":
        # 将配置文件拷贝一份放到输出目录中
        shutil.copy2(args.config_path, (model_dir / "config.yaml").as_posix())

# make logger
    logger = get_logger("", log_file=Path(model_dir / f"{args.mode}.log").as_posix())
    # pkg_version = check_version(cfg.get("joeynmt_version", None))
    logger.info("Hello! This is Joey-NMT (version %s).", "666")

    if args.use_ddp:
        n_gpu = torch.cuda.device_count() \
            if cfg.get("use_cuda", False) and torch.cuda.is_available() else 0
        if args.mode == "train":
            assert n_gpu > 1, "gpu数量不对，得大于1"
            logger.info(f"开始多卡并行{n_gpu}")
            cfg["use_ddp"] = args.use_ddp
            mp.spawn(train,args=(n_gpu, cfg, args.skip_test), nprocs=n_gpu)
        elif args.mode == "test":
            raise RuntimeError("测试模式，DDP无法使用")
        elif args.mode == "translate":
            raise RuntimeError(
                "翻译模型，DDP无法使用"
            )
    else:
        if args.mode == "train":
            train(rank=0, world_size=None, cfg=cfg, skip_test=args.skip_test)



if __name__ == "__main__":
    main()
