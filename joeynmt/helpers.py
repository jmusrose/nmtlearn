import shutil
from pathlib import Path





def make_model_dir(model_dir:Path, overwrite:bool=False):
    """
    创建一个模型的目录
    :param model_dir:
    :param overwrite:
    :return:
    """
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite:
            raise FileExistsError(f"{model_dir}已经存在")

        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
