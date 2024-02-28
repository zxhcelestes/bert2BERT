from .gpt_mindspore import GPT2Model
from .loader_mindspore import set_decoder


def expand_gpt(org_gpt, target_gpt_config, method):
    """
    :param org_gpt: 预训练好的小gpt
    :param target_gpt_config: 大gpt的参数日志
    :param method: 扩展策略
    :return: 大gpt
    """
    if target_gpt_config.size_per_head is not None:
        assert target_gpt_config.size_per_head == org_gpt.size_per_head

    new_gpt = GPT2Model(target_gpt_config, is_training=True)
    decoder = []
    # 找到decoder块
    modules = org_gpt.name_cells()
    for key in modules.keys():
        if "gpt2_decoder" in key:
            decoder.append(modules.get(key))

    modules = new_gpt.name_cells()
    for key in modules.keys():
        if "gpt2_decoder" in key:
            decoder.append(modules.get(key))
    set_decoder(new_gpt, org_gpt, decoder[0], decoder[1], org_hidden_size=org_gpt.n_embed,
                target_hidden_size=new_gpt.n_embed,
                method=method,
                new_num_layers=new_gpt.n_layer)
    return new_gpt
