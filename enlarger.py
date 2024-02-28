import os.path

from get_path import root
from src.Bert_Model.mindspore_version.load_pretrain_model import load_bert_base as ld_ms, enlarge as enlarge_ms, \
    pre_defined_bert_config as config_ms
from src.Bert_Model.pytorch_version.load_pretrain_model import load_bert_base as ld_py, enlarge as enlarge_py, \
    pre_defined_bert_config as config_py


class PytorchEnlarger:

    @staticmethod
    def bert_base2large_aki():
        ckpt_path = f"{root}/ckpt/bert_base.ckpt"
        save_path = f"{root}/output/bert_base2large_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "bert_base", load_gitee_ckpt=True, specify_prefix="bert.bert")
            new_model = enlarge_py(model, config_py("bert_large"), "AKI", save_path)
            return new_model

    @staticmethod
    def bert_base2large_fpi():
        ckpt_path = f"{root}/ckpt/bert_base.ckpt"
        save_path = f"{root}/output/bert_base2large_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "bert_base", load_gitee_ckpt=True, specify_prefix="bert.bert")
            new_model = enlarge_py(model, config_py("bert_large"), "FPI", save_path)
            return new_model

    @staticmethod
    def bert_small2base_aki():
        ckpt_path = f"{root}/ckpt/bert_small.ckpt"
        save_path = f"{root}/output/bert_small2base_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "bert_small", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("bert_base"), "AKI", save_path)
            return new_model

    @staticmethod
    def bert_small2base_fpi():
        ckpt_path = f"{root}/ckpt/bert_small.ckpt"
        save_path = f"{root}/output/bert_small2base_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "bert_small", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("bert_base"), "FPI", save_path)
            return new_model


class MindsporeEnlarger:

    @staticmethod
    def bert_base2large_aki():
        ckpt_path = f"{root}/ckpt/bert_base.ckpt"
        save_path = f"{root}/output/bert_base2large_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "bert.cls1"],
                                    specify_prefix="bert.bert", kind="bert_base", load_gitee_ckpt=True)
            new_model = enlarge_ms(pre_train_model, config_ms("bert_large"), "AKI", save_path)
            return new_model

    @staticmethod
    def bert_base2large_fpi():
        ckpt_path = f"{root}/ckpt/bert_base.ckpt"
        save_path = f"{root}/output/bert_base2large_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, filter_prefix=["lamb_m", "bert.cls1"],
                                    specify_prefix="bert.bert", kind="bert_base", load_gitee_ckpt=True)
            new_model = enlarge_ms(pre_train_model, config_ms("bert_large"), "FPI", save_path)
            return new_model

    @staticmethod
    def bert_small2base_aki():
        ckpt_path = f"{root}/ckpt/bert_small.ckpt"
        save_path = f"{root}/output/bert_small2base_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, kind="bert_small", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("bert_base"), "AKI", save_path)
            return new_model

    @staticmethod
    def bert_small2base_fpi():
        ckpt_path = f"{root}/ckpt/bert_small.ckpt"
        save_path = f"{root}/output/bert_small2base_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, kind="bert_small", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("bert_base"), "FPI", save_path)
            return new_model
