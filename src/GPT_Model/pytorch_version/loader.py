import torch.nn
from loguru import logger

from .expand import expand_fpi, expand_aki, expand_copy, expand_aki_copy, generate_random_match
from .find_utils import find_ffn_block, find_mha_block, find_embeddings, find_dense_weight, find_number


def set_block1(new_model, org_block, org_hidden_size, target_hidden_size):
    """
    根据论文附录图13，从下往上第一个模块 。这个模块和wpe/wte链接，所以设计Wemb，后续decoder不需要操作这个参数
    :param new_model: 新模型
    :param org_block: [W_emb, W_ln, W_l^{QKV}]
    :param org_hidden_size: 初始隐藏层规模
    :param target_hidden_size: 目标隐藏层规模
    """
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    head_size = target_hidden_size * 3

    for (key, param) in org_block:
        weights = param
        if "wpe" in key or "wte" in key:
            if weights.ndim == 2:
                m, n = weights.shape
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=m,
                                       target_col=target_hidden_size, to_expand="col")
            elif weights.ndim == 1:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                       target_col=1, to_expand="row")
            else:
                raise Exception("维度超过2")
        elif "c_attn" in key:
            # KQV三个矩阵
            # 对注意力模型内部参数的修改
            if weights.ndim == 2:
                dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                      target_col=head_size, to_expand="row")
            elif weights.ndim == 1:
                # 截距项
                dic[key] = torch.concat([weights, torch.zeros(head_size - weights.shape[0])], 0)
                # dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=head_size,
                #                       target_col=1, to_expand="row")
            else:
                raise Exception
        else:
            raise Exception("块1中不存在该层")

        if key in dic.keys():
            logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")

    return dic


def set_block2(new_model, org_block, org_hidden_size, target_hidden_size):
    """
    根据论文附录图13，从下往上第二个块 。这个模块有W_o W_LN W_l1
    :param new_model: 新模型
    :param org_block: [W_o W_LN W_l1]
    :param org_hidden_size: 初始隐藏层规模
    :param target_hidden_size: 目标隐藏层规模
    """
    head_size = target_hidden_size * 3
    intermediate = new_model.config.intermediate_size
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    for key, param in org_block:
        weights = param
        if "attn.c_proj" in key:
            if weights.ndim == 2:
                temp = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                  target_col=target_hidden_size, to_expand="row")
                dic[key] = expand_copy(temp, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                       target_col=target_hidden_size, to_expand="col")
            elif weights.ndim == 1:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                       target_col=1, to_expand="row")
            else:
                raise Exception("维度超过2")


        elif "ln_1" in key:
            dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                   target_col=1, to_expand="row")
        # W_l^1
        # 前馈网络部分参数修改，有两个Liner层
        elif "mlp.c_fc" in key:
            if "weight" in key:
                dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                      target_col=intermediate, to_expand="row")
            elif "bias" in key:
                # 扩展bias在FFN扩展部分。此处不需要操作,复制即可
                dic[key] = torch.concat([weights, torch.zeros(intermediate - weights.shape[0])], 0)
                # dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=intermediate,
                #                       target_col=1, to_expand="row")
            else:
                raise Exception
        else:
            raise Exception(f"块2中不存在{key}层")
        if key in dic.keys():
            logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")
    return dic


def set_block3(new_model, org_block, org_hidden_size, target_hidden_size):
    """
    根据论文附录图13，从下往上第三个块 。有第二个前馈层，以及归一层
    :param new_model: 新模型
    :param org_block: W_l2 W_LN
    :param org_hidden_size: 初始隐藏层规模
    :param target_hidden_size: 目标隐藏层规模
    """
    intermediate = new_model.config.intermediate_size
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    for key, param in org_block:
        weights = param
        # gpt2_decoder.0.output.LayerNorm.weight --> torch.Size([108])
        # gpt2_decoder.0.output.LayerNorm.bias --> torch.Size([108])
        if "mlp.c_proj" in key:
            if "weight" in key:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=intermediate,
                                       target_col=target_hidden_size, to_expand="col")
            elif "bias" in key:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                       target_col=1, to_expand="row")
            else:
                raise Exception
        elif "ln_2" in key:
            if weights.ndim == 1:
                dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                      target_col=1, to_expand="row")
            else:
                raise Exception
        else:
            raise Exception(f"块3中不存在{key}层")
        logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")
    return dic


def set_ffn_fpi(new_model, ffn_block, org_intermediate_size, target_intermediate_size):
    """
    FPI策略扩展FFN结构
    :param new_model: 新模型
    :param ffn_block: 带扩展的FFN块
    :param org_intermediate_size: 初始FFN中间层规模
    :param target_intermediate_size: 目标FFN中间层规模
    """
    choose_num_dict = generate_random_match(org_intermediate_size, target_intermediate_size)
    hidden_size = new_model.config.hidden_size
    dic = dict()
    for key, param in ffn_block:
        weights = param
        if "c_fc" in key:
            if "weight" in key:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=hidden_size,
                                       target_col=target_intermediate_size, to_expand="col")
            elif "bias" in key:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_intermediate_size,
                                       target_col=1,
                                       to_expand="row")
            else:
                raise Exception
        elif "c_proj" in key:
            if "weight" in key:
                dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_intermediate_size,
                                      target_col=hidden_size, to_expand="row")
            elif "bias" in key:
                # dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=hidden_size,
                #                        target_col=1, to_expand="row")
                dic[key] = torch.concat([weights, torch.zeros(hidden_size - weights.shape[0])], 0)

            else:
                raise Exception
        else:
            raise Exception
        logger.info(f"FFN_FPI: expand {key}: ")
    return dic


def set_ffn_aki(new_model, ffn_block, ffn_block_nxt, org_intermediate_size, target_intermediate_size):
    """
    AKI策略扩展FFN结构
    :param new_model: 新模型
    :param ffn_block: 待扩展的FFN块
    :param ffn_block_nxt: 下一层对应的FFN块
    :param org_intermediate_size: 初始FFN中间层规模
    :param target_intermediate_size: 目标FFN中间层规模
    """
    choose_num_dict = generate_random_match(org_intermediate_size, target_intermediate_size)
    hidden_size = new_model.config.hidden_size
    dic = dict()
    for (key, param), (key_nxt, param_nxt) in zip(ffn_block, ffn_block_nxt):
        weights = param
        if "c_fc" in key:
            if "weight" in key:
                dic[key] = expand_aki_copy(weights, param_nxt, choose_num_dict=choose_num_dict,
                                           target_row=hidden_size,
                                           target_col=target_intermediate_size, to_expand="col")
            elif "bias" in key:
                dic[key] = expand_aki_copy(weights, param_nxt, choose_num_dict=choose_num_dict,
                                           target_row=target_intermediate_size,
                                           target_col=1,
                                           to_expand="row")
            else:
                raise Exception
        elif "c_proj" in key:
            if "weight" in key:
                dic[key] = expand_aki(weights, param_nxt, choose_num_dict=choose_num_dict,
                                      target_row=target_intermediate_size,
                                      target_col=hidden_size, to_expand="row")
            elif "bias" in key:
                # dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_intermediate_size,
                #                       target_col=1,
                #                       to_expand="row")
                dic[key] = torch.concat([weights, torch.zeros(hidden_size - weights.shape[0])], 0)
            else:
                raise Exception
        else:
            raise Exception
        if key in dic.keys():
            logger.info(f"FFN_AKI: use {key_nxt} expand {key}: ")
    return dic


def set_mha_fpi(new_model, mha_block, org_head_size, head_size):
    """
    FPI策略扩展多头
    :param new_model: 新模型
    :param mha_block: 待扩展的MHA块
    :param org_head_size: 初始头数目
    :param head_size: 目标头数目
    """
    choose_head_dict = generate_random_match(org_head_size, head_size)

    dic = dict()
    for key, param in mha_block:
        weights = param
        if "c_attn" in key:
            # 扩展输出，只需要copy
            if "weight" in key:
                dic[key] = expand_copy(weights, target_row=new_model.config.hidden_size, target_col=head_size,
                                       choose_num_dict=choose_head_dict, to_expand="col")
            elif "bias" in key:
                dic[key] = expand_copy(weights, target_row=head_size, target_col=1,
                                       choose_num_dict=choose_head_dict, to_expand="row")
            else:
                raise Exception
        if key in dic.keys():
            logger.info(f"MHA_FPI: expand {key}: ")
    return dic


def set_mha_aki(new_model, mha_block, mha_block_nxt, org_head_size, head_size):
    """
    AKI策略扩展多头
    :param new_model: 新模型
    :param mha_block: 多头块
    :param mha_block_nxt: 下一层多头块
    :param org_head_size: 原始头数目
    :param head_size: 目标头数目
    """
    # 选择的头 (10:1) 十号头选1号头的参数
    choose_head_dict = generate_random_match(org_head_size, head_size)

    dic = dict()
    for (key, param), (key_nxt, param_nxt) in zip(mha_block, mha_block_nxt):
        weights = param
        weights_nxt = param_nxt
        if "c_attn" in key:
            # 扩展输出，只需要copy
            if "weight" in key:
                dic[key] = expand_aki_copy(weights, weights_nxt, target_row=new_model.config.hidden_size,
                                           target_col=head_size,
                                           choose_num_dict=choose_head_dict, to_expand="col")
            elif "bias" in key:
                dic[key] = expand_aki_copy(weights, weights_nxt, target_row=head_size, target_col=1,
                                           choose_num_dict=choose_head_dict, to_expand="row")
            else:
                raise Exception
        if key in dic.keys():
            logger.info(f"MHA_AKI: use {key_nxt} expand {key}: ")
    return dic


def set_decoder(new_model, org_model, org_decoder, new_decoder, org_hidden_size, target_hidden_size,
                new_num_layers=None,
                method="FPI"):
    """
    修改decoder参数规模,decoder 一般是一个ModuleList，包含多个GPTLayer，每个GPTLayer有如下结构
    :param new_model: 扩展后的模型
    :param org_model: 原始模型
    :param org_decoder: GPT中decoder块
    :param org_hidden_size: 初始隐藏层规模
    :param target_hidden_size: 目标隐藏层规模
    :param new_num_layers: 目标decoder层数
    :param method: 方法(FPI/AKI)
    """
    decoder_layers = org_decoder.named_children()
    # 获取layers的ModuleList
    modulelist = list()
    for i in decoder_layers:
        modulelist.append(i[1])
    # modulelist = layers[0][1]
    org_layers = org_model.config.num_hidden_layers

    # 获取ModuleList中每一个 GPTDecoderCell

    # 最后一层，只能用FPI
    if modulelist.__len__() == 1:
        method = "FPI"
    # step1 每个layer都跑一遍更新步骤1，不管AKI还是FPI
    logger.critical(f"step1 开始: 对{org_model.config.num_hidden_layers}个decoder。分三个块进行FPI")

    for i in range(org_model.config.num_hidden_layers):
        temp_layer = modulelist.__getitem__(i)
        set_GPT_layer_fpi(new_model, org_model, temp_layer, org_hidden_size, target_hidden_size, level=i,
                          prefix=f"gpt2_decoder.{i}.")

    # 完成step1后，用新的矩阵完成后续扩展
    decoder_layers = new_decoder.named_children()
    # 获取layers的ModuleList
    modulelist = list()
    for i in decoder_layers:
        modulelist.append(i[1])
    # modulelist = layers[0][1]
    org_layers = org_model.config.num_hidden_layers

    # step2 FFN扩展和MHA扩展(如果需要的话)
    logger.critical(f"step2 开始: 使用{method}策略扩展FFN或MHA(如果需要的话)")
    # FFN扩展
    if new_model.config.intermediate_size > org_model.config.intermediate_size:
        logger.critical("FFN扩展开始")
        dic = dict()
        if method == "FPI":
            for i in range(org_layers):
                temp_layer = modulelist.__getitem__(i)
                ffn_block = find_ffn_block(temp_layer, f"gpt2_decoder.{i}.")
                temp_dic = set_ffn_fpi(new_model, ffn_block, org_intermediate_size=org_model.config.intermediate_size,
                                       target_intermediate_size=new_model.config.intermediate_size)
                dic.update(temp_dic)
        elif method == "AKI":
            temp_layer = modulelist.__getitem__(0)
            for i in range(org_layers - 1):
                nxt_layer = modulelist.__getitem__(i + 1)
                ffn_block = find_ffn_block(temp_layer, f"gpt2_decoder.{i}.")
                ffn_block_nxt = find_ffn_block(nxt_layer, f"gpt2_decoder.{i + 1}.")
                temp_dict = set_ffn_aki(new_model, ffn_block, ffn_block_nxt, org_model.config.intermediate_size,
                                        new_model.config.intermediate_size)
                temp_layer = nxt_layer
                dic.update(temp_dict)
            ffn_block = find_ffn_block(temp_layer, f"gpt2_decoder.{org_layers - 1}.")
            temp_dict = set_ffn_fpi(new_model, ffn_block, org_model.config.intermediate_size,
                                    new_model.config.intermediate_size)
            dic.update(temp_dict)

        else:
            raise Exception
        # 参数load进模型
        logger.info("导入参数")
        logger.info(dic.keys())
        new_model.load_state_dict(dic, strict=False)

    # 多头扩展
    if new_model.config.num_attention_heads > org_model.config.num_attention_heads:
        logger.critical("MHA扩展开始")
        dic = dict()
        if method == "FPI":
            for i in range(org_layers):
                temp_layer = modulelist.__getitem__(i)
                mha_block = find_mha_block(temp_layer, f"gpt2_decoder.{i}.")
                head_size = new_model.config.hidden_size * 3
                org_head_size = org_model.config.hidden_size * 3
                temp_dic = set_mha_fpi(new_model, mha_block, org_head_size, head_size)
                dic.update(temp_dic)
        elif method == "AKI":
            temp_layer = modulelist.__getitem__(0)
            for i in range(org_layers - 1):
                nxt_layer = modulelist.__getitem__(i + 1)
                mha_block = find_mha_block(temp_layer, f"gpt2_decoder.{i}.")
                mha_block_nxt = find_mha_block(nxt_layer, f"gpt2_decoder.{i + 1}.")
                head_size = new_model.config.hidden_size * 3
                org_head_size = org_model.config.hidden_size * 3
                temp_dict = set_mha_aki(new_model, mha_block, mha_block_nxt,
                                        org_head_size, head_size)
                temp_layer = nxt_layer
                dic.update(temp_dict)
            mha_block = find_mha_block(temp_layer, f"gpt2_decoder.{org_layers - 1}.")
            head_size = new_model.config.hidden_size * 3
            org_head_size = org_model.config.hidden_size * 3
            temp_dict = set_mha_fpi(new_model, mha_block, org_head_size, head_size)
            dic.update(temp_dict)

        else:
            raise Exception
        # 参数load进模型
        logger.info("导入参数")
        logger.info(dic.keys())
        new_model.load_state_dict(dic, strict=False)

    # 深度扩展。论文的Algorithm 1
    if new_num_layers is None or org_layers == new_num_layers:
        pass
    else:
        # 计算是否能够整除？不能整除，则n！=0
        k, n = new_num_layers // org_layers, new_num_layers % org_layers
        logger.critical(f"深度扩展开始:纵向复制{k - 1}次，高位补齐{n}位")
        # 找到新模型中已经修改好的decoder_block
        paras_dict = new_model.state_dict()
        # 将Decoder前org_num_layers的GPTlayer提取，组成decoder_block
        decoder_block = dict()
        for layer_name in paras_dict.keys():
            if "decoder" in layer_name:
                if int(find_number(layer_name)) < org_layers:
                    # 组成decoder_block
                    decoder_block[layer_name] = paras_dict.get(layer_name)

        for i in range(1, k):
            start = i * org_layers
            end = start + org_layers
            set_depth(new_model, decoder_block, org_layers, start, end)
        # 多余的n层，用高处几层填上
        set_depth(new_model, decoder_block, org_layers, k * org_layers, k * org_layers + n)
    # 把分类器的参数扩展
    logger.critical("开始扩展分类器参数")
    dense_dict = set_dense(new_model, org_model, org_hidden_size, target_hidden_size)
    new_model.load_state_dict(dense_dict, strict=False)


def set_depth(new_model, decoder_block, num_layers, start_idx, end_idx):
    """
    进行深度方向上的decoder块堆叠
    :param new_model: 新模型
    :param decoder_block: 已经完成AKI或者FPI的有org_num_layers层的decoder块,是一个state_dict
    :param num_layers: decoder_block的层数
    :param start_idx: 待堆叠的layer下标
    :param end_idx: 尾部layer层下标
    """
    temp_dict = dict()
    for idx in range(start_idx, end_idx):
        # equal为idx对应的低层layer层

        # 补足的部分，用相对top的参数
        if end_idx - start_idx != num_layers:
            equal = idx % num_layers + num_layers - (end_idx - start_idx)
        else:
            equal = idx % num_layers
        # 复制低layer层的参数，更改名称，放进当前的dict
        for name in decoder_block.keys():
            num_lay = find_number(name)
            if str(equal) == num_lay:
                temp_name = name.replace("." + str(equal) + ".", "." + str(idx) + ".", 1)
                logger.info(f"{name}-->{temp_name}")
                layer = decoder_block.get(name)
                temp_dict[temp_name] = layer.data
    new_model.load_state_dict(temp_dict, strict=False)


def set_GPT_layer_fpi(new_model, org_model, GPT_layer, org_hidden_size, target_hidden_size, level, prefix):
    """
    GPT_layer: 定义好的GPTLayer结构
    :param new_model: 新模型
    :param org_model: 原始模型
    :param GPT_layer: 待操作的1个decoder
    :param org_hidden_size: 初始隐藏规模
    :param target_hidden_size: 目标隐藏规模
    :param level: decoder级数
    :param prefix: 前缀
    """
    all_layers = list(GPT_layer.named_parameters())
    block1 = []
    block2 = []
    block3 = []
    # 在第一级的decoder，需要加入embedding_table
    for name, param in all_layers:
        name = prefix + name
        if "c_attn" in name:
            block1.append((name, param))
        elif "attn.c_proj" in name or "ln_1" in name or "mlp.c_fc" in name:
            block2.append((name, param))
        elif "mlp.c_proj" in name or "ln_2" in name:
            block3.append((name, param))

    if level == 0:
        embeddings = find_embeddings(org_model)
        block1 = embeddings + block1
    # for info in block1:
    #     key = info[0]
    #     print(key)
    dic1 = set_block1(new_model, block1, org_hidden_size, target_hidden_size)
    dic2 = set_block2(new_model, block2, org_hidden_size, target_hidden_size)
    dic3 = set_block3(new_model, block3, org_hidden_size, target_hidden_size)
    dic1.update(dic2)
    dic1.update(dic3)
    logger.critical("导入参数")
    logger.info(dic1.keys())
    warn = new_model.load_state_dict(dic1, strict=False)


def set_dense(new_model, org_model, org_hidden_size, target_hidden_size):
    """
    扩展分类器参数
    :param new_model: 新模型
    :param org_model: W_l2 W_LN
    :param org_hidden_size: 初始隐藏层规模
    :param target_hidden_size: 目标隐藏层规模
    """
    dense_block = find_dense_weight(org_model)
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    for key, weights in dense_block:
        if "weight" in key:
            dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                   target_col=1, to_expand="row")
        elif "bias" in key:
            dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                   target_col=1, to_expand="row")
        else:
            raise Exception

        logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")
    return dic
