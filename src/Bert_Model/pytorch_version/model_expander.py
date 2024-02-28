from .huggingface_bert import BertModel
from .loader import set_encoder


def expand_bert(org_bert, target_bert_config, method):
    """

    :param org_bert: pre trained little bert
    :param target_bert_config: parameter log of large bert
    :param method: extension policy
    :return: Big bert
    """
    if target_bert_config.size_per_head is not None:
        assert target_bert_config.size_per_head == org_bert.config.size_per_head

    new_bert = BertModel(target_bert_config)
    encoder = []
    # Encoder block found
    modules = org_bert.named_children()
    for key, data in modules:
        if "encoder" in key:
            encoder.append(data)
    modules = new_bert.named_children()
    for key, data in modules:
        if "encoder" in key:
            encoder.append(data)
    org_enc = encoder[0]
    new_enc = encoder[1]
    set_encoder(new_bert, org_bert, org_enc, new_enc, org_hidden_size=org_bert.config.hidden_size,
                target_hidden_size=new_bert.config.hidden_size,
                method=method,
                new_num_layers=new_bert.config.num_hidden_layers)

    return new_bert
