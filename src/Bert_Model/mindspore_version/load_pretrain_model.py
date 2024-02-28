import mindspore
import mindspore.common.dtype as mstype
from .bert_config import BertConfig
from .bert_mindspore import BertModel
from .model_expander_mindspore import expand_bert


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
                          dtype=mstype.float32,
                          compute_type=mstype.float32)

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
                          dtype=mstype.float32,
                          compute_type=mstype.float32)
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
                          dtype=mstype.float32,
                          compute_type=mstype.float32)

    else:
        raise Exception("This model setting is not included")


def load_bert_base(path, kind, filter_prefix=None, specify_prefix=None, load_gitee_ckpt=False):
    params = mindspore.load_checkpoint(path, filter_prefix=filter_prefix,
                                       specify_prefix=specify_prefix)
    params_dict = dict()
    if load_gitee_ckpt:
        for key in params.keys():
            # rename
            new_key = ".".join(key.split(".")[2:])
            params_dict[new_key] = params.get(key)
    else:
        params_dict = params

    bert = pre_defined_bert_config(kind)
    model = BertModel(bert, is_training=True)

    info = mindspore.load_param_into_net(model, params_dict, strict_load=True)

    if len(info) == 0:
        return model
    else:
        raise Exception("Model parameters are not imported completely.")


def enlarge(model, target_config, method, save_path):
    new_model = expand_bert(model, target_config, method)
    mindspore.save_checkpoint(new_model, save_path)
    return new_model
