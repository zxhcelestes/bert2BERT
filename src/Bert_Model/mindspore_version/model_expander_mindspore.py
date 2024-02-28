from .bert_mindspore import BertModel
from .loader_mindspore import set_encoder


def expand_bert(org_bert, target_bert_config, method):
    """
    :param org_bert: pre trained little bert
    :param target_bert_config: parameter log of large bert
    :param method: extension policy
    :return: Big bert
    """
    if target_bert_config.size_per_head is not None:
        assert target_bert_config.size_per_head == org_bert.size_per_head

    new_bert = BertModel(target_bert_config, is_training=True)
    encoder = []
    # Encoder block found
    modules = org_bert.name_cells()
    for key in modules.keys():
        if "encoder" in key:
            encoder.append(modules.get(key))

    modules = new_bert.name_cells()
    for key in modules.keys():
        if "encoder" in key:
            encoder.append(modules.get(key))
    set_encoder(new_bert, org_bert, encoder[0], encoder[1], org_hidden_size=org_bert.hidden_size,
                target_hidden_size=new_bert.hidden_size,
                method=method,
                new_num_layers=new_bert.num_hidden_layers)
    return new_bert
