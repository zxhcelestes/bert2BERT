import argparse

import mindspore as ms
import torch

from logger import logger
from src.GPT_Model.mindspore_version.load_pretrain_model import load_gpt_base as load_ms
from tokenizer import get_sst_tokenizer


def _infer_mindspore(sentence, ms_gpt_path, kind):
    """
    gpt reasoning in mindspore
    :param sentence: indicates a statement
    :param ms_gpt_path: ckpt file of mindspore gpt
    :param kind: indicates the type of gpt
    :return: embedding
    """
    ms_gpt = load_ms(ms_gpt_path, kind)
    # Get the teleprompter
    tokenizer = get_sst_tokenizer()
    inputs = tokenizer(sentence, return_tensors="pt")
    # complement 128 bits
    for key in inputs:
        inputs[key] = torch.concat([inputs[key], torch.zeros(1, 128 - inputs[key].shape[1], dtype=inputs[key].dtype)],
                                   1)
    inputs["input_mask"] = inputs.get("attention_mask")
    inputs.pop("attention_mask")
    # Turn torch into mindspore
    for key in inputs:
        inputs[key] = ms.Tensor.from_numpy(inputs[key].numpy()).squeeze()
    sequence_output, pooled_output, embedding_tables = ms_gpt.construct(**inputs)
    ms_output = pooled_output.asnumpy()
    logger.info(f"The statement input is {sentence}")
    logger.info("ms->{}".format(ms_output))
    return ms_output


def infer_ms():
    parser = argparse.ArgumentParser(description="vocab embedding infer")
    parser.add_argument('--sentence', type=str, default="",
                        help="sentence")
    parser.add_argument('--ms_gpt_path', type=str, help="gpt ckpt file")
    parser.add_argument('--kind', type=str, choices=["gpt_base", "gpt_large", "gpt_small"], help="gpt kind")
    args = parser.parse_args()
    print(args)
    _infer_mindspore(args.sentence, args.ms_gpt_path, args.kind)


if __name__ == '__main__':
    infer_ms()
