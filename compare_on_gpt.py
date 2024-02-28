import mindspore as ms
import numpy as np
import torch

from logger import logger
from tokenizer_on_gpt import get_sst_tokenizer


def compare_model_weights(ms_gpt, py_gpt):
    """
    Compare the Mindspore and Pytorch versions of gpt parameters
    :param ms_gpt: Mindspore gpt_Model
    :param py_gpt: Pytorch gpt_Model
    :return: error value
    """
    ms_dict = list(ms_gpt.get_parameters())
    pyt_dict = list(py_gpt.named_parameters())

    diff = 0
    for x, y in zip(ms_dict, pyt_dict):
        print(x.name, "<--->", y[0])
        ms_weights = x.value().asnumpy()
        py_weights = y[1].detach().numpy()
        delta = np.sum(np.abs(ms_weights - py_weights))
        print("error value is: ", delta)
        diff += np.sum(delta)
    return diff


def compare_vocab_embedding(sentence, ms_gpt, py_gpt):
    """
    Compare SST data set under the embedding model given after the input sentence
    :param sentence: sentence
    :param ms_gpt: mindspore gpt
    :param py_gpt: pytorch gpt
    """
    # 获取提词器
    tokenizer = get_sst_tokenizer()
    inputs = tokenizer(sentence, return_tensors="pt")
    # 补足1024位
    for key in inputs:
        inputs[key] = torch.concat([inputs[key], torch.zeros(1, 1024 - inputs[key].shape[1], dtype=inputs[key].dtype)],
                                   1)
    # 打开测试模式，去除dropout的影响
    py_gpt.eval()
    py_output = py_gpt(**inputs).last_hidden_state.detach().numpy()
    inputs["input_mask"] = inputs.get("attention_mask")
    inputs.pop("attention_mask")
    # 把torch变成mindspore
    for key in inputs:
        inputs[key] = ms.Tensor.from_numpy(inputs[key].numpy())
    ms_gpt.batch_size = 1
    decoder_output, embedding_tables = ms_gpt.construct(**inputs)
    ms_output = decoder_output.asnumpy()
    logger.critical("Begin testing the embedding representation of the SST data set！！！")
    logger.info(f"The statement input is{sentence}")
    logger.critical("The output of the two gpt is：")
    logger.info("py->{}".format(py_output))
    logger.info("ms->{}".format(ms_output))

    logger.critical("embedding error value is{}".format(np.sum(np.abs((ms_output - py_output)))))
