import os.path

from get_path import root
from src.GPT_Model.mindspore_version.load_pretrain_model import load_gpt_base as ld_ms, enlarge as enlarge_ms, \
    pre_defined_gpt_config as config_ms
from src.GPT_Model.pytorch_version.load_pretrain_model import load_gpt_base as ld_py, enlarge as enlarge_py, \
    pre_defined_gpt_config as config_py


class PytorchEnlarger:

    @staticmethod
    def gpt_large2xl_aki():
        ckpt_path = f"{root}/ckpt/gpt_large.ckpt"
        save_path = f"{root}/output/gpt_large2xl_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_large", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_xl"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_large2xl_fpi():
        ckpt_path = f"{root}/ckpt/gpt_large.ckpt"
        save_path = f"{root}/output/gpt_large2xl_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_large", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_xl"), "FPI", save_path)
            return new_model

    @staticmethod
    def gpt_medium2large_aki():
        ckpt_path = f"{root}/ckpt/gpt_medium.ckpt"
        save_path = f"{root}/output/gpt_medium2large_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_medium", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_large"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_medium2large_fpi():
        ckpt_path = f"{root}/ckpt/gpt_medium.ckpt"
        save_path = f"{root}/output/gpt_medium2large_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_medium", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_large"), "FPI", save_path)
            return new_model

    @staticmethod
    def gpt_base2medium_aki():
        ckpt_path = f"{root}/ckpt/gpt_base.ckpt"
        save_path = f"{root}/output/gpt_base2medium_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_base", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_medium"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_base2medium_fpi():
        ckpt_path = f"{root}/ckpt/gpt_base.ckpt"
        save_path = f"{root}/output/gpt_base2medium_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "gpt_base", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("gpt_medium"), "FPI", save_path)
            return new_model


class MindsporeEnlarger:

    @staticmethod
    def gpt_large2xl_aki():
        ckpt_path = f"{root}/ckpt/gpt_large.ckpt"
        save_path = f"{root}/output/gpt_large2xl_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "gpt.cls1"],
                                    specify_prefix=None, kind="gpt_large", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_xl"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_large2xl_fpi():
        ckpt_path = f"{root}/ckpt/gpt_large.ckpt"
        save_path = f"{root}/output/gpt_large2xl_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "gpt.cls1"],
                                    specify_prefix=None, kind="gpt_large", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_xl"), "FPI", save_path)
            return new_model

    @staticmethod
    def gpt_medium2large_aki():
        ckpt_path = f"{root}/ckpt/gpt_medium.ckpt"
        save_path = f"{root}/output/gpt_medium2large_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "gpt.cls1"],
                                    specify_prefix=None, kind="gpt_medium", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_large"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_medium2large_fpi():
        ckpt_path = f"{root}/ckpt/gpt_medium.ckpt"
        save_path = f"{root}/output/gpt_medium2large_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "gpt.cls1"],
                                    specify_prefix=None, kind="gpt_medium", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_large"), "FPI", save_path)
            return new_model

    @staticmethod
    def gpt_base2medium_aki():
        ckpt_path = f"{root}/ckpt/gpt_base.ckpt"
        save_path = f"{root}/output/gpt_base2medium_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, kind="gpt_base", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_medium"), "AKI", save_path)
            return new_model

    @staticmethod
    def gpt_base2medium_fpi():
        ckpt_path = f"{root}/ckpt/gpt_base.ckpt"
        save_path = f"{root}/output/gpt_base2medium_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, kind="gpt_base", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("gpt_medium"), "FPI", save_path)
            return new_model
