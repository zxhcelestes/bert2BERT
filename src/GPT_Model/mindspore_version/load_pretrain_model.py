import mindspore
import mindspore.common.dtype as mstype

from .gpt_mindspore import GPT2Model
from .GPT_config import GPT2Config
from .model_expander_mindspore import expand_gpt


def pre_defined_gpt_config(name):
    if name == "gpt_base":
        return GPT2Config(batch_size=1,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embed=768,
                          n_layer=12,
                          n_head=12,
                          intermediate_size=3072,
                          hidden_act="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=mstype.float32,
                          compute_type=mstype.float16)

    elif name == "gpt_medium":
        return GPT2Config(batch_size=1,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embed=1024,
                          n_layer=24,
                          n_head=16,
                          intermediate_size=3072,
                          hidden_act="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=mstype.float32,
                          compute_type=mstype.float16)

    elif name == "gpt_large":
        return GPT2Config(batch_size=1,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embed=1280,
                          n_layer=36,
                          n_head=20,
                          intermediate_size=3072,
                          hidden_act="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=mstype.float32,
                          compute_type=mstype.float16)

    elif name == "gpt_xl":
        return GPT2Config(batch_size=1,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embed=1600,
                          n_layer=48,
                          n_head=25,
                          intermediate_size=3072,
                          hidden_act="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=mstype.float32,
                          compute_type=mstype.float16)
    else:
        raise Exception("未包含该模型设定")


def load_gpt_base(path, kind, filter_prefix=None, specify_prefix=None, load_gitee_ckpt=False):
    params = mindspore.load_checkpoint(path, filter_prefix=filter_prefix,
                                       specify_prefix=specify_prefix)
    params_dict = dict()
    if load_gitee_ckpt:
        for key in params.keys():
            # 重命名
            new_key = ".".join(key.split(".")[2:])
            params_dict[new_key] = params.get(key)
    else:
        params_dict = params

    gpt = pre_defined_gpt_config(kind)
    model = GPT2Model(gpt, is_training=True)

    info = mindspore.load_param_into_net(model, params_dict, strict_load=True)

    if len(info) == 0:
        return model
    else:
        raise Exception("模型参数未导入完全。")


def enlarge(model, target_config, method, save_path):
    new_model = expand_gpt(model, target_config, method)
    mindspore.save_checkpoint(new_model, save_path)
    return new_model
