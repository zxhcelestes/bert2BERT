from .GPT_config import GPT2Config
from .huggingface_gpt import GPT2Model
from .loader import set_decoder


def expand_gpt(org_gpt, target_gpt_config, method="FPI"):
    """
    :param org_gpt: 预训练好的小gpt
    :param target_gpt_config: 大gpt的参数日志
    :param method: 扩展策略
    :return: 大gpt
    """
    if target_gpt_config.size_per_head is not None:
        assert target_gpt_config.size_per_head == org_gpt.config.size_per_head

    new_gpt = GPT2Model(target_gpt_config)
    encoder = []
    # 找到Encoder块
    modules = org_gpt.named_children()
    for key, data in modules:
        if "gpt2_decoder" in key:
            encoder.append(data)
    modules = new_gpt.named_children()
    for key, data in modules:
        if "gpt2_decoder" in key:
            encoder.append(data)
    org_enc = encoder[0]
    new_enc = encoder[1]
    set_decoder(new_gpt, org_gpt, org_enc, new_enc, org_hidden_size=org_gpt.config.hidden_size,
                target_hidden_size=new_gpt.config.hidden_size,
                method=method,
                new_num_layers=new_gpt.config.num_hidden_layers)

    return new_gpt


if __name__ == '__main__':
    org_model = GPT2Model(GPT2Config(num_hidden_layers=12, hidden_size=120))
    new_model = GPT2Model(GPT2Config(vocab_size=28996, num_hidden_layers=24, hidden_size=240))
    # 预训练gpt
    # org_model = gptModel.from_pretrained("gpt-base-cased")
    expand_gpt(org_gpt=org_model, target_gpt_config=GPT2Config(num_hidden_layers=24, hidden_size=240))
