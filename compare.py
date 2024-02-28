import mindspore as ms
import numpy as np
import torch

from logger import logger
from tokenizer import get_sst_tokenizer


def compare_model_weights(ms_bert, py_bert):
    """
    Compare the Bert parameters of Mindspot and Python versions
    :param ms_bert: Mindspore Bert_Model
    :param py_bert: Pytorch Bert_Model
    :return: Error value
    """
    ms_dict = list(ms_bert.get_parameters())
    pyt_dict = list(py_bert.named_parameters())
    # The position of one parameter is reversed. Replace it
    pyt_dict[1], pyt_dict[2] = pyt_dict[2], pyt_dict[1]
    diff = 0
    for x, y in zip(ms_dict, pyt_dict):
        print(x.name, "<--->", y[0])
        ms_weights = x.value().asnumpy()
        py_weights = y[1].detach().numpy()
        delta = np.sum(np.abs(ms_weights - py_weights))
        print("The error is: ", delta)
        diff += np.sum(delta)
    return diff


def compare_vocab_embedding(sentence, ms_bert, py_bert):
    """
    Compare the embedding given by the model after entering sentences in the SST dataset
    :param sentence: 语句
    :param ms_bert: mindspore的bert
    :param py_bert: pytorch的bert
    """
    # Get Prompter
    tokenizer = get_sst_tokenizer()
    inputs = tokenizer(sentence, return_tensors="pt")
    # Complement 128
    for key in inputs:
        inputs[key] = torch.concat([inputs[key], torch.zeros(1, 128 - inputs[key].shape[1], dtype=inputs[key].dtype)],
                                   1)
    # Open the test mode to remove the impact of dropout
    py_bert.eval()
    py_output = py_bert(**inputs).pooler_output.detach().numpy()

    inputs["input_mask"] = inputs.get("attention_mask")
    inputs.pop("attention_mask")
    # Turn torch into mindspot
    for key in inputs:
        inputs[key] = ms.Tensor.from_numpy(inputs[key].numpy()).squeeze()
    sequence_output, pooled_output, embedding_tables = ms_bert.construct(**inputs)
    ms_output = pooled_output.asnumpy()
    logger.critical("Start the SST dataset embedding test!!!")
    logger.info(f"Sentence input is{sentence}")
    logger.critical("The outputs of the two Berts are:")
    logger.info("py->{}".format(py_output))
    logger.info("ms->{}".format(ms_output))

    logger.critical("embedding error is {}".format(np.sum(np.abs((ms_output - py_output)))))
    return
