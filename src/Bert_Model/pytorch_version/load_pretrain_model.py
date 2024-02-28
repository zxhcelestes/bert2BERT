import mindspore as ms
import torch

from .huggingface_bert import BertModel, BertConfig
from .model_expander import expand_bert


def pre_defined_bert_config(name):
    if name == "bert_base":
        return BertConfig(seq_length=128,
                          vocab_size=21128,
                          hidden_size=768,
                          num_hidden_layers=12,
                          num_attention_heads=12,
                          size_per_head=768 // 12,
                          intermediate_size=3072,
                          hidden_act="gelu",
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          max_position_embeddings=512,
                          type_vocab_size=2,
                          initializer_range=0.02,
                          use_relative_positions=False,
                          dtype=torch.float32,
                          compute_type=torch.float32)

    elif name == "bert_large":
        return BertConfig(seq_length=128,
                          vocab_size=21128,
                          hidden_size=1024,
                          num_hidden_layers=24,
                          num_attention_heads=16,
                          size_per_head=1024 // 16,
                          intermediate_size=3072,
                          hidden_act="gelu",
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          max_position_embeddings=512,
                          type_vocab_size=2,
                          initializer_range=0.02,
                          use_relative_positions=False,
                          dtype=torch.float32,
                          compute_type=torch.float32)

    elif name == "bert_small":
        return BertConfig(seq_length=128,
                          vocab_size=21128,
                          hidden_size=512,
                          num_hidden_layers=4,
                          num_attention_heads=8,
                          size_per_head=512 // 8,
                          intermediate_size=2048,
                          hidden_act="gelu",
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          max_position_embeddings=512,
                          type_vocab_size=2,
                          initializer_range=0.02,
                          use_relative_positions=False,
                          dtype=torch.float32,
                          compute_type=torch.float32)
    else:
        raise Exception("This model setting is not included")


def trans_ms_2_pytorch(param):
    # to numpy
    weights = param.value().asnumpy()
    return [param.name, torch.tensor(weights, dtype=torch.float32, requires_grad=True)]


def pytorch_load_ckpt(path):
    ckpt_dict = ms.load_checkpoint(path)
    new_dict = dict()
    for key in ckpt_dict.keys():
        new_dict[key] = trans_ms_2_pytorch(ckpt_dict.get(key))
    return new_dict


def load_bert_base(path, kind, filter_prefix=None, specify_prefix=None, load_gitee_ckpt=False):
    params = ms.load_checkpoint(path, filter_prefix=filter_prefix,
                                specify_prefix=specify_prefix)
    params_dict = dict()
    if load_gitee_ckpt:
        for key in params.keys():
            # rename
            new_key = ".".join(key.split(".")[2:])
            params_dict[new_key] = trans_ms_2_pytorch(params.get(key))
    else:
        params_dict = dict()
        for key in params.keys():
            params_dict[key] = trans_ms_2_pytorch(params.get(key))

    bert_config = pre_defined_bert_config(kind)
    model = BertModel(bert_config)

    pyt_dict = model.state_dict()
    position_ids = pyt_dict.get('embeddings.position_ids')
    # This has no pre training parameters
    pyt_dict.pop('embeddings.position_ids')
    org_keys = list(params_dict.keys())
    new_keys = list(pyt_dict.keys())
    # The position of one parameter is reversed. Replace it
    new_keys[1], new_keys[2] = new_keys[2], new_keys[1]
    for x, y in zip(org_keys, new_keys):
        pyt_dict[y] = params_dict.get(x)[1]

    pyt_dict['embeddings.position_ids'] = position_ids
    model.load_state_dict(state_dict=pyt_dict, strict=True)
    return model


def enlarge(model, target_config, method, save_path):
    new_model = expand_bert(model, target_config, method)
    torch.save(new_model.state_dict(), save_path)
    return new_model
