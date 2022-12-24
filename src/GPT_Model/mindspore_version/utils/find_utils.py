import re


def find_ffn_block(bert_layer):
    all_layers = list(bert_layer.get_parameters())
    ffn_block = []
    for param in all_layers:
        name = param.name
        if "feedforward.c_fc" in name or "feedforward.c_proj.weight" in name:
            ffn_block.append(param)
    return ffn_block


def find_mha_block(bert_layer):
    all_layers = list(bert_layer.get_parameters())
    mha_block = []

    for param in all_layers:
        name = param.name
        if "masked_self_attention.c_attn" in name:
            mha_block.append(param)
    return mha_block


def find_embeddings(new_model):
    all_layers = list(new_model.get_parameters())
    embeddings = []
    for param in all_layers:
        name = param.name
        if "embedding" in name:
            embeddings.append(param)
        else:
            break
    return embeddings


def find_dense_weight(new_model):
    dense_part = []
    pattern = "^layer_norm.*"
    all_layers = list(new_model.get_parameters())
    for param in all_layers:
        name = param.name
        lst = re.findall(pattern, name)
        if len(lst) > 0:
            dense_part.append(param)
    return dense_part


def find_number(string):
    lst = re.findall("\.\d+\.", string)
    if len(lst) != 1:
        raise Exception
    number = lst[0].strip(".")
    return number
