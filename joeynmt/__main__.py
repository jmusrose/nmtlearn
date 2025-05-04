import argparse
from joeynmt.config import load_config
from pathlib import Path
from joeynmt.helpers import make_model_dir

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


if __name__ == "__main__":
    main()
