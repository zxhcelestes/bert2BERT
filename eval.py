import argparse
import random
import time

import torch

from compare import compare_model_weights, compare_vocab_embedding
from enlarger import PytorchEnlarger, MindsporeEnlarger
from logger import logger
from src.Bert_Model.mindspore_version.load_pretrain_model import load_bert_base as load_ms
from src.Bert_Model.pytorch_version.huggingface_bert import BertModel
from src.Bert_Model.pytorch_version.load_pretrain_model import pre_defined_bert_config


class Comparator:
    sentence = "Hello, my dog is cute."
    timecost = []

    def test_small2base_fpi(self):
        random.seed(10)
        start = time.time()
        PytorchEnlarger.bert_small2base_fpi()
        self.timecost.append(time.time() - start)

        random.seed(10)
        start = time.time()
        MindsporeEnlarger.bert_small2base_fpi()
        self.timecost.append(time.time() - start)
        logger.critical(f"Python extension time{self.timecost[-2]}s，MindSpore expansion time {self.timecost[-1]}s")
        # pytorch
        large_config = pre_defined_bert_config("bert_base")
        pytorch_model = BertModel(large_config)
        pytorch_model.load_state_dict(torch.load("output/bert_small2base_fpi.pth"))

        # mindspore
        ms_model = load_ms(path="output/bert_small2base_fpi.ckpt", specify_prefix=None, kind="bert_base")
        logger.info("-" * 40)
        logger.info(
            "small2base_fpi model parameter error is:{} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)

    def test_base2large_fpi(self):
        random.seed(10)
        start = time.time()
        PytorchEnlarger.bert_base2large_fpi()
        self.timecost.append(time.time() - start)

        random.seed(10)
        start = time.time()
        MindsporeEnlarger.bert_base2large_fpi()
        self.timecost.append(time.time() - start)
        logger.critical(f"Python extension time{self.timecost[-2]}s，MindSpore expansion time {self.timecost[-1]}s")
        # pytorch
        large_config = pre_defined_bert_config("bert_large")
        pytorch_model = BertModel(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/bert_base2large_fpi.pth"))

        # mindspore
        ms_model = load_ms(path="output/bert_base2large_fpi.ckpt", specify_prefix=None, kind="bert_large")
        logger.info("-" * 40)
        logger.info("base2large_fpi parameter error is: {} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)

    def test_base2large_aki(self):
        random.seed(10)
        start = time.time()
        PytorchEnlarger.bert_base2large_aki()
        self.timecost.append(time.time() - start)

        random.seed(10)
        start = time.time()
        MindsporeEnlarger.bert_base2large_aki()
        self.timecost.append(time.time() - start)
        logger.critical(f"Python extension time{self.timecost[-2]}s，MindSpore expansion time {self.timecost[-1]}s")
        # pytorch
        large_config = pre_defined_bert_config("bert_large")
        pytorch_model = BertModel(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/bert_base2large_aki.pth"))

        # mindspore
        ms_model = load_ms(path="output/bert_base2large_aki.ckpt", specify_prefix=None, kind="bert_large")
        logger.info("-" * 40)
        logger.info("base2large_akiparameter error is: {} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)

    def test_small2base_aki(self):
        random.seed(10)
        start = time.time()
        PytorchEnlarger.bert_small2base_aki()
        self.timecost.append(time.time() - start)

        random.seed(10)
        start = time.time()
        MindsporeEnlarger.bert_small2base_aki()
        self.timecost.append(time.time() - start)
        logger.critical(f"Python extension time{self.timecost[-2]}s，MindSpore expansion time {self.timecost[-1]}s")
        # pytorch
        large_config = pre_defined_bert_config("bert_base")
        pytorch_model = BertModel(large_config)
        # print(pytorch_model)
        pytorch_model.load_state_dict(torch.load("output/bert_small2base_aki.pth"))

        # mindspore
        ms_model = load_ms(path="output/bert_small2base_aki.ckpt", specify_prefix=None, kind="bert_base")
        logger.info("-" * 40)
        logger.info("small2base_aki parameter error is: {} ".format(compare_model_weights(ms_model, pytorch_model)))
        compare_vocab_embedding(self.sentence, ms_model, pytorch_model)
        logger.info("-" * 40)


def run_eval():
    """ evaluate scripts """
    parser = argparse.ArgumentParser(description="Bert_Model expansion")
    parser.add_argument('--task_type', type=str, default="", choices=["small2base", "base2large"],
                        help="expansion type.")
    parser.add_argument('--method', type=str, default="fpi", choices=["fpi", "aki"], help="Expansion strategy.")
    comparator = Comparator()
    args = parser.parse_args()
    if args.task_type == "small2base":
        if args.method == "fpi":
            comparator.test_small2base_fpi()
        elif args.method == "aki":
            comparator.test_small2base_aki()
        else:
            raise Exception("invalid args")
    elif args.task_type == "base2large":
        if args.method == "fpi":
            comparator.test_base2large_fpi()
        elif args.method == "aki":
            comparator.test_base2large_aki()
        else:
            raise Exception("invalid args")
    else:
        raise Exception("invalid args")


if __name__ == '__main__':
    run_eval()
