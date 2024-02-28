import argparse
import os.path

import mindspore

from get_path import root
from src.GPT_Model.mindspore_version.load_pretrain_model import load_gpt_base as ld_ms, enlarge as enlarge_ms, \
    pre_defined_gpt_config as config_ms


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


def run_eval():
    """ evaluate scripts """
    parser = argparse.ArgumentParser(description="gpt_Model expansion")
    parser.add_argument('--task_type', type=str, default="", choices=["base2medium", "medium2large", "large2xl"],
                        help="expansion type.")
    parser.add_argument('--method', type=str, default="fpi", choices=["fpi", "aki"], help="Expansion strategy.")
    parser.add_argument('--device_target', type=str, choices=["CPU", "Ascend"], default="CPU", help="run device")

    enlarger_ms = MindsporeEnlarger()
    args = parser.parse_args()
    mindspore.set_context(device_target=args.device_target)
    if args.task_type == "base2medium":
        if args.method == "fpi":
            enlarger_ms.gpt_base2medium_fpi()
        elif args.method == "aki":
            enlarger_ms.gpt_base2medium_aki()
        else:
            raise Exception("invalid args")
    elif args.task_type == "medium2large":
        if args.method == "fpi":
            enlarger_ms.gpt_medium2large_fpi()
        elif args.method == "aki":
            enlarger_ms.gpt_base2medium_aki()
        else:
            raise Exception("invalid args")
    elif args.task_type == "large2xl":
        if args.method == "fpi":
            enlarger_ms.gpt_large2xl_fpi()
        elif args.method == "aki":
            enlarger_ms.gpt_large2xl_aki()
        else:
            raise Exception("invalid args")
    else:
        raise Exception("invalid args")


if __name__ == '__main__':
    run_eval()
