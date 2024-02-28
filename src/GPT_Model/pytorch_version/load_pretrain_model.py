import mindspore as ms
import torch

from .GPT_config import GPT2Config
from .huggingface_gpt import GPT2Model
from .model_expander import expand_gpt


def pre_defined_gpt_config(name):
    if name == "gpt_base":
        return GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=768,
                          n_layer=12,
                          n_head=12,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)

    elif name == "gpt_medium":
        return GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=1024,
                          n_layer=24,
                          n_head=16,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)

    elif name == "gpt_large":
        return GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=1280,
                          n_layer=36,
                          n_head=20,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)

    elif name == "gpt_xl":
        return GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=1600,
                          n_layer=48,
                          n_head=25,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)
    else:
        raise Exception("未包含该模型设定")


def trans_ms_2_pytorch(param):
    # 转numpy
    weights = param.value().asnumpy()
    return [param.name, torch.tensor(weights, dtype=torch.float32, requires_grad=True)]


def pytorch_load_ckpt(path):
    ckpt_dict = ms.load_checkpoint(path)
    new_dict = dict()
    for key in ckpt_dict.keys():
        new_dict[key] = trans_ms_2_pytorch(ckpt_dict.get(key))
    return new_dict


# 如果是生成的ckpt那么load_gitee_ckpt置为False，specify_prefix根据具体情况赋值
def load_gpt_base(path, kind, filter_prefix=None, specify_prefix=None, load_gitee_ckpt=False):
    params = ms.load_checkpoint(path, filter_prefix=filter_prefix,
                                specify_prefix=specify_prefix)
    params_dict = dict()
    if load_gitee_ckpt:
        for key in params.keys():
            # 重命名
            new_key = ".".join(key.split(".")[2:])
            params_dict[new_key] = trans_ms_2_pytorch(params.get(key))
    else:
        params_dict = dict()
        for key in params.keys():
            params_dict[key] = trans_ms_2_pytorch(params.get(key))

    GPT_config = pre_defined_gpt_config(kind)
    model = GPT2Model(GPT_config)

    pyt_dict = model.state_dict()
    # print(len(pyt_dict.keys()))
    # quit()
    org_keys = list(params_dict.keys())
    new_keys = list()
    tmp = 0
    for name, param in model.named_parameters():
        new_keys.append(name)
    for x, y in zip(org_keys, new_keys):
        # print(x, "     ", y)
        # print(params_dict.get(x)[1].shape, "     ", pyt_dict.get(y).shape)
        pyt_dict[y] = params_dict.get(x)[1]
    # pyt_dict['embeddings.position_ids'] = position_ids
    model.load_state_dict(state_dict=pyt_dict, strict=True)
    return model


def enlarge(model, target_config, method, save_path):
    new_model = expand_gpt(model, target_config, method)
    torch.save(new_model.state_dict(), save_path)
    return new_model
