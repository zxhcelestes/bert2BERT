import argparse
import os.path

import mindspore

from get_path import root
from src.Bert_Model.mindspore_version.load_pretrain_model import load_bert_base as ld_ms, enlarge as enlarge_ms, \
    pre_defined_bert_config as config_ms


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


def run_eval():
    """ evaluate scripts """
    parser = argparse.ArgumentParser(description="Bert_Model expansion")
    parser.add_argument('--task_type', type=str, default="", choices=["small2base", "base2large"],
                        help="expansion type.")
    parser.add_argument('--method', type=str, default="fpi", choices=["fpi", "aki"], help="Expansion strategy.")
    parser.add_argument('--device_target', type=str, choices=["CPU", "Ascend"], default="CPU", help="run device")
    enlarger_ms = MindsporeEnlarger()
    args = parser.parse_args()
    mindspore.set_context(device_target=args.device_target)
    if args.task_type == "small2base":
        if args.method == "fpi":
            enlarger_ms.bert_small2base_fpi()
        elif args.method == "aki":
            enlarger_ms.bert_small2base_aki()
        else:
            raise Exception("invalid args")
    elif args.task_type == "base2large":
        if args.method == "fpi":
            enlarger_ms.bert_base2large_fpi()
        elif args.method == "aki":
            enlarger_ms.bert_small2base_aki()
        else:
            raise Exception("invalid args")
    else:
        raise Exception("invalid args")


if __name__ == '__main__':
    run_eval()
