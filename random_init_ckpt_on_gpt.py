import mindspore
from mindspore.common.initializer import initializer, Normal

from src.GPT_Model.mindspore_version.gpt_mindspore import GPT2Model
from src.GPT_Model.mindspore_version.load_pretrain_model import pre_defined_gpt_config


def random_init_ckpt(path, kind):
    """
    gpt parameters of kind type are generated randomly
    :param path: save the path
    param kind: specifies the gpt specification, such as gpt_base
    """
    gpt_config = pre_defined_gpt_config(kind)
    model = GPT2Model(gpt_config, is_training=True)
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
    path = "./ckpt/gpt_medium"
    random_init_ckpt(path, "gpt_medium")
