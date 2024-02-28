import mindspore
from mindspore.common.initializer import initializer, Normal

from src.Bert_Model.mindspore_version.bert_mindspore import BertModel
from src.Bert_Model.mindspore_version.load_pretrain_model import pre_defined_bert_config


def random_init_ckpt(path, kind):
    """
    The Bert parameter of kind type is randomly generated
    :param path: save the path
    :param kind: Bert size, such as bert_small
    """
    bert_config = pre_defined_bert_config(kind)
    model = BertModel(bert_config, is_training=True)
    dic = dict()
    for param in model.get_parameters():
        name = param.name
        shape = param.shape
        dic[name] = initializer(Normal(), shape, mindspore.float32)
    # Parameters are loaded into the model
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    mindspore.load_param_into_net(model, dic, strict_load=False)
    mindspore.save_checkpoint(model, path)


if __name__=='__main__':
    path = "./ckpt/bert_small"
    random_init_ckpt(path, "bert_small")
