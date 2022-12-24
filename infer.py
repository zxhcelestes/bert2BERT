import argparse

import mindspore as ms
import torch

from logger import logger
from src.Bert_Model.mindspore_version.load_pretrain_model import load_bert_base as load_ms
from tokenizer import get_sst_tokenizer


def _infer_mindspore(sentence, ms_bert_path, kind):
    """
    mindspore bert infer
    :param sentence
    :param ms_bert_path: The ckpt file of mindspore bert
    :param kind: bert type
    :return: embedding
    """

    ms_bert = load_ms(ms_bert_path, kind)
    tokenizer = get_sst_tokenizer()
    inputs = tokenizer(sentence, return_tensors="pt")
    # Complement 128
    for key in inputs:
        inputs[key] = torch.concat([inputs[key], torch.zeros(1, 128 - inputs[key].shape[1], dtype=inputs[key].dtype)],
                                   1)
    inputs["input_mask"] = inputs.get("attention_mask")
    inputs.pop("attention_mask")
    # Turn torch into mindspore
    for key in inputs:
        inputs[key] = ms.Tensor.from_numpy(inputs[key].numpy()).squeeze()
    sequence_output, pooled_output, embedding_tables = ms_bert.construct(**inputs)
    ms_output = pooled_output.asnumpy()
    logger.info(f"Sentence input is {sentence}")
    logger.info("ms->{}".format(ms_output))
    return ms_output


def infer_ms():
    parser = argparse.ArgumentParser(description="vocab embedding infer")
    parser.add_argument('--sentence', type=str, default="",
                        help="sentence")
    parser.add_argument('--ms_bert_path', type=str, help="bert ckpt file")
    parser.add_argument('--kind', type=str, choices=["bert_base", "bert_large", "bert_small"], help="bert kind")
    args = parser.parse_args()
    print(args)
    _infer_mindspore(args.sentence, args.ms_bert_path, args.kind)


if __name__ == '__main__':
    infer_ms()
