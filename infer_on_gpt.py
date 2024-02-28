import argparse

import mindspore as ms
import numpy as np

from logger import logger
from src.GPT_Model.mindspore_version.load_pretrain_model import load_gpt_base as load_ms
from tokenizer_on_gpt import get_sst_tokenizer


def _infer_mindspore_gpt(sentence, ms_gpt_path, kind):
    """
    gpt reasoning in mindspore
    :param sentence: indicates a statement
    :param ms_gpt_path: ckpt file of mindspore gpt
    :param kind: indicates the type of gpt
    :return: embedding
    """
    ms_gpt = load_ms(ms_gpt_path, kind)
    ms_gpt.batch_size = 1
    # Get the teleprompter
    tokenizer = get_sst_tokenizer()
    inputs = tokenizer(sentence, return_tensors="np")
    # complement 1024 bits
    for key in inputs:
        inputs[key] = np.concatenate([inputs[key], np.zeros([1, 1024 - inputs[key].shape[1]], dtype=np.int16)], axis=1)
    inputs["input_mask"] = inputs.get("attention_mask")
    inputs.pop("attention_mask")
    for key in inputs:
        inputs[key] = ms.Tensor.from_numpy(inputs[key]).squeeze()
    decoder_output, embedding_tables = ms_gpt.construct(**inputs)
    ms_output = decoder_output.asnumpy()
    logger.info(f"The statement input is {sentence}")
    logger.critical("The output of the two gpt is：")
    logger.info("ms->{}".format(ms_output))
    return ms_output


def infer_ms():
    parser = argparse.ArgumentParser(description="vocab embedding infer")
    parser.add_argument('--sentence', type=str, default="",
                        help="sentence")
    parser.add_argument('--ms_gpt_path', type=str, help="gpt ckpt file")
    parser.add_argument('--kind', type=str, choices=["gpt_base", "gpt_large", "gpt_small","gpt_medium"], help="gpt kind")
    parser.add_argument('--device_target', type=str, choices=["CPU", "Ascend"], help="run device")

    args = parser.parse_args()
    ms.set_context(device_target=args.device_target)
    print(args)
    _infer_mindspore_gpt(args.sentence, args.ms_gpt_path, args.kind)


if __name__ == '__main__':
    infer_ms()
