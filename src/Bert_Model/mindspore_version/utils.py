import re


def find_ffn_block(bert_layer):
    all_layers = list(bert_layer.get_parameters())
    ffn_block = []
    for param in all_layers:
        name = param.name
        if "intermediate" in name or ("output.dense" in name and "attention" not in name):
            ffn_block.append(param)
    return ffn_block


def find_mha_block(bert_layer):
    all_layers = list(bert_layer.get_parameters())
    mha_block = []

    for param in all_layers:
        name = param.name
        if "key" in name or "query" in name or "value" in name or ("output.dense" in name and "attention" in name):
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
    pattern = "^dense.*"
    all_layers = list(new_model.get_parameters())
    for param in all_layers:
        name = param.name
        lst = re.findall(pattern, name)
        if len(lst) > 0:
            dense_part.append(param)
    return dense_part


def find_number(string):
    lst = re.findall(r"\.\d+\.", string)
    if len(lst) != 1:
        raise Exception
    number = lst[0].strip(".")
    return number
