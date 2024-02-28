import re


def find_ffn_block(bert_layer, prefix):
    all_layers = list(bert_layer.named_parameters())
    ffn_block = []
    for name, param in all_layers:
        if "mlp.c_fc" in name or "mlp.c_proj" in name:
            ffn_block.append((prefix + name, param))
    return ffn_block


def find_mha_block(bert_layer, prefix):
    all_layers = list(bert_layer.named_parameters())
    mha_block = []

    for name, param in all_layers:
        if "c_attn" in name:
            mha_block.append((prefix + name, param))
    return mha_block


def find_embeddings(new_model):
    all_layers = list(new_model.named_parameters())
    embeddings = []
    for name, param in all_layers:
        if "wte" in name or "wpe" in name:
            embeddings.append((name, param))
        else:
            break
    return embeddings


def find_dense_weight(new_model):
    dense_part = []
    pattern = "^ln_f.*"
    all_layers = list(new_model.named_parameters())
    for name, param in all_layers:
        lst = re.findall(pattern, name)
        if len(lst) > 0:
            dense_part.append((name, param))
    return dense_part


def find_number(string):
    lst = re.findall("\.\d+\.", string)
    if len(lst) != 1:
        raise Exception
    number = lst[0].strip(".")
    return number
