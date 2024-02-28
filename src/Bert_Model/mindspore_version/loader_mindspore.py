import mindspore
from loguru import logger

from .expand_mindspore import expand_fpi, expand_aki, expand_copy, expand_aki_copy, \
    generate_random_match
from .load_into_net_mindspore import load_param_into_net
from .utils import *


def set_block1(new_model, org_block, org_hidden_size, target_hidden_size):
    """
    According to Figure 13 in the appendix of the paper, the first module is from bottom to top.
    This module is linked to embedding, so Wemb is designed,
    and subsequent encoders do not need to operate this parameter
    :param new_model: new model
    :param org_block: [W_emb, W_ln, W_l^{QKV}]
    :param org_hidden_size: initial hidden layer size
    :param target_hidden_size: target hidden layer size
    """
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    head_size = new_model.size_per_head * new_model.num_attention_heads

    for param in org_block:
        key = param.name
        weights = param
        if "embedding" in key:
            if weights.ndim == 2:
                m, n = weights.shape
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=m,
                                       target_col=target_hidden_size, to_expand="col")
            elif weights.ndim == 1:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                       target_col=1, to_expand="row")
            else:
                raise Exception("Dimension exceeds 2")
        elif "attention" in key:
            # KQV three matrices
            # Modification of internal parameters of attention model
            if weights.ndim == 2:
                dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=head_size,
                                      target_col=target_hidden_size, to_expand="col")
            elif weights.ndim == 1:
                # bias
                concat = mindspore.ops.Concat(0)
                zeros = mindspore.ops.Zeros()
                dic[key] = concat([weights, zeros((target_hidden_size - weights.shape[0]), mindspore.float32)])

            else:
                raise Exception
        else:
            raise Exception("This layer does not exist in block 1")

        if key in dic.keys():
            logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")
    # Parameter load input model
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    return dic


def set_block2(new_model, org_block, org_hidden_size, target_hidden_size):
    """
    According to Figure 13 in the appendix of the paper, the second block is from bottom to top. T
    his module has W_ o W_ LN W_ l1
    :param new_model: new model
    :param org_block: [W_o W_LN W_l1]
    :param org_hidden_size: initial hidden layer size
    :param target_hidden_size: target hidden layer size
    """
    head_size = new_model.num_attention_heads * new_model.size_per_head
    intermediate = new_model.intermediate_size
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    for param in org_block:
        key = param.name
        weights = param
        if "attention" in key:
            if "output" in key:
                if weights.ndim == 2:
                    dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                           target_col=head_size, to_expand="row")
                elif weights.ndim == 1:
                    dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                           target_col=1, to_expand="row")
                else:
                    raise Exception("Dimension exceeds 2")
            else:
                raise Exception("Parameters other than output should not appear")
        # W_l^1
        # Some parameters of feedforward network are modified, and there are two liner layers
        elif "intermediate" in key:
            if "weight" in key:
                dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=intermediate,
                                      target_col=target_hidden_size, to_expand="col")
            elif "bias" in key:
                # Expand the bios in the FFN extension part. No action required here
                concat = mindspore.ops.Concat(0)
                zeros = mindspore.ops.Zeros()
                dic[key] = concat([weights, zeros((intermediate - weights.shape[0]), mindspore.float32)])
            else:
                raise Exception
        else:
            raise Exception(f"{key} layer does not exist in block 2")
        if key in dic.keys():
            logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    return dic


def set_block3(new_model, org_block, org_hidden_size, target_hidden_size):
    """
    According to Figure 13 in the appendix of the paper, the third block is from bottom to top.
    There is a second feedforward layer and a unified layer
    :param new_model: new model
    :param org_block: W_ l2 W_ LN
    :param org_hidden_size: initial hidden layer size
    :param target_hidden_size: target hidden layer size
    """
    intermediate = new_model.intermediate_size
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    for param in org_block:
        key = param.name
        weights = param
        if "output" in key:
            if "dense" in key:
                if "weight" in key:
                    dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                           target_col=intermediate, to_expand="row")
                elif "bias" in key:
                    dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                           target_col=1, to_expand="row")
                else:
                    raise Exception
            elif "layernorm" in key:
                if weights.ndim == 1:
                    dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                          target_col=1, to_expand="row")
                else:
                    raise Exception
        else:
            raise Exception(f"{key} layer does not exist in block 3")
        logger.info(f"{key}:  {weights.shape}  -->  {dic.get(key).shape}")
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    return dic


def set_ffn_fpi(new_model, ffn_block, org_intermediate_size, target_intermediate_size):
    """
    FPI policy extends FFN structure
    :param new_model: new model
    :param ffn_block: FFN block with extension
    :param org_intermediate_size: Initial FFN M-server size
    :param target_intermediate_size: Target FFN M-server size
    """
    choose_num_dict = generate_random_match(org_intermediate_size, target_intermediate_size)
    hidden_size = new_model.hidden_size
    dic = dict()
    for param in ffn_block:
        key = param.name
        weights = param
        if "intermediate" in key:
            if "weight" in key:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_intermediate_size,
                                       target_col=hidden_size, to_expand="row")
            elif "bias" in key:
                dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_intermediate_size,
                                       target_col=1,
                                       to_expand="row")
            else:
                raise Exception
        elif "output.dense" in key:
            if "weight" in key:
                dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=hidden_size,
                                      target_col=target_intermediate_size, to_expand="col")
            elif "bias" in key:
                concat = mindspore.ops.Concat(0)
                zeros = mindspore.ops.Zeros()
                dic[key] = concat([weights, zeros((hidden_size - weights.shape[0]), mindspore.float32)])
            else:
                raise Exception
        else:
            raise Exception
        logger.info(f"FFN_FPI: expand {key}: ")
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    return dic


def set_ffn_aki(new_model, ffn_block, ffn_block_nxt, org_intermediate_size, target_intermediate_size):
    """
    AKI policy extends FFN structure
    :param new_model: new model
    :param ffn_block: FFN block to be extended
    :param ffn_block_nxt: FFN block corresponding to the next layer
    :param org_intermediate_size: Initial FFN M-server size
    :param target_intermediate_size: Target FFN M-server size
    """
    choose_num_dict = generate_random_match(org_intermediate_size, target_intermediate_size)
    hidden_size = new_model.hidden_size
    dic = dict()
    for param, param_nxt in zip(ffn_block, ffn_block_nxt):
        key = param.name
        key_nxt = param_nxt.name
        weights = param
        if "intermediate" in key:
            if "weight" in key:
                dic[key] = expand_aki_copy(weights, param_nxt, choose_num_dict=choose_num_dict,
                                           target_row=target_intermediate_size,
                                           target_col=hidden_size, to_expand="row")
            elif "bias" in key:
                dic[key] = expand_aki_copy(weights, param_nxt, choose_num_dict=choose_num_dict,
                                           target_row=target_intermediate_size,
                                           target_col=1,
                                           to_expand="row")
            else:
                raise Exception
        elif "output.dense" in key:
            if "weight" in key:
                dic[key] = expand_aki(weights, param_nxt, choose_num_dict=choose_num_dict, target_row=hidden_size,
                                      target_col=target_intermediate_size, to_expand="col")
            elif "bias" in key:
                concat = mindspore.ops.Concat(0)
                zeros = mindspore.ops.Zeros()
                dic[key] = concat([weights, zeros((hidden_size - weights.shape[0]), mindspore.float32)])
            else:
                raise Exception
        else:
            raise Exception
        if key in dic.keys():
            logger.info(f"FFN_AKI: use {key_nxt} expand {key}: ")
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    return dic


def set_mha_fpi(new_model, mha_block, org_head_num, target_head_num):
    """
    FPI policy extension header
    :param new_model: new model
    :param mha_block: MHA block to be extended
    :param org_head_num: initial number of headers
    :param target_head_num: number of target headers
    """
    # Selected head (10:1) Parameters for selecting head 1 from head 10
    size_per_head = new_model.size_per_head
    num_heads = new_model.num_attention_heads
    head_size = size_per_head * num_heads
    choose_head_dict = generate_random_match(org_head_num, target_head_num)
    new_dict = dict()
    # Extend the set to fit the input requirements
    for key in choose_head_dict.keys():
        # Parameter block of the selected header
        chosen_span = [*range(choose_head_dict.get(key) * size_per_head,
                              choose_head_dict.get(key) * size_per_head + size_per_head)]
        # Parameter block of the selected header
        choose_span = [*range(key * size_per_head, key * size_per_head + size_per_head)]
        for after, pre in zip(choose_span, chosen_span):
            new_dict[after] = pre

    dic = dict()
    for param in mha_block:
        key = param.name
        weights = param
        if "query" in key or "key" in key or "value" in key:
            # Extended output, only copy is required
            if "weight" in key:
                dic[key] = expand_copy(weights, target_row=head_size, target_col=new_model.hidden_size,
                                       choose_num_dict=new_dict, to_expand="row")
            elif "bias" in key:
                dic[key] = expand_copy(weights, target_row=head_size, target_col=1,
                                       choose_num_dict=new_dict, to_expand="row")
            else:
                raise Exception
        elif "output" in key:
            # Extended input, fpi required
            if "weight" in key:
                dic[key] = expand_fpi(weights, target_row=new_model.hidden_size, target_col=head_size,
                                      choose_num_dict=new_dict, to_expand="col")
            elif "bias" in key:
                concat = mindspore.ops.Concat(0)
                zeros = mindspore.ops.Zeros()
                dic[key] = concat([weights, zeros((new_model.hidden_size - weights.shape[0]), mindspore.float32)])
            else:
                raise Exception
        if key in dic.keys():
            logger.info(f"MHA_FPI: expand {key}: ")
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    return dic


def set_mha_aki(new_model, mha_block, mha_block_nxt, org_head_num, target_head_num):
    """
    AKI policy extension header
    :param new_model: new model
    :param mha_block: multi header block
    :param mha_block_nxt: next layer multi header block
    :param org_head_num: number of original headers
    :param target_head_num: number of target headers
    """
    size_per_head = new_model.size_per_head
    num_heads = new_model.num_attention_heads
    head_size = size_per_head * num_heads
    choose_head_dict = generate_random_match(org_head_num, target_head_num)
    new_dict = dict()
    for key in choose_head_dict.keys():
        chosen_span = [*range(choose_head_dict.get(key) * size_per_head,
                              choose_head_dict.get(key) * size_per_head + size_per_head)]
        choose_span = [*range(key * size_per_head, key * size_per_head + size_per_head)]
        for after, pre in zip(choose_span, chosen_span):
            new_dict[after] = pre
    dic = dict()
    for param, param_nxt in zip(mha_block, mha_block_nxt):
        key = param.name
        key_nxt = param_nxt.name
        weights = param
        weights_nxt = param_nxt
        if "query" in key or "key" in key or "value" in key:
            if "weight" in key:
                dic[key] = expand_aki_copy(weights, weights_nxt, target_row=head_size, target_col=new_model.hidden_size,
                                           choose_num_dict=new_dict, to_expand="row")
            elif "bias" in key:
                dic[key] = expand_aki_copy(weights, weights_nxt, target_row=head_size, target_col=1,
                                           choose_num_dict=new_dict, to_expand="row")
            else:
                raise Exception
        elif "output" in key:
            # Extension input, aki required
            if "weight" in key:
                dic[key] = expand_aki(weights, weights_nxt, target_row=new_model.hidden_size, target_col=head_size,
                                      choose_num_dict=new_dict, to_expand="col")
            elif "bias" in key:
                concat = mindspore.ops.Concat(0)
                zeros = mindspore.ops.Zeros()
                dic[key] = concat([weights, zeros((new_model.hidden_size - weights.shape[0]), mindspore.float32)])
            else:
                raise Exception
        if key in dic.keys():
            logger.info(f"MHA_AKI: use {key_nxt} expand {key}: ")
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    return dic


def set_encoder(new_model, org_model, org_encoder, new_encoder, org_hidden_size, target_hidden_size,
                new_num_layers=None,
                method="FPI"):
    """
    Modify the encoder parameter scale. The encoder is generally a ModuleList, including multiple BertLayer.
     Each BertLayer has the following structure
    :param new_model: extended model
    :param org_model: original model
    :param org_encoder: encoder block in bert
    :param org_hidden_size: initial hidden layer size
    :param target_hidden_size: target hidden layer size
    :param new_num_layers: target encoder layers
    :param method: method (FPI/AKI)
    """
    encoder_layers = org_encoder.name_cells()
    # Get the ModuleList of layers
    layers = list(encoder_layers.values())
    modulelist = layers[0]

    # Get every BertEncoderCell in ModuleList

    # The last layer can only use FPI
    if modulelist.__len__() == 1:
        method = "FPI"

    # Step 1 Run the update step 1 for each layer.
    # Just reduce the depth of the model. Whether AKI or FPI
    logger.critical(f"Step 1 starts: {org_model.num_hidden_layers} encoders. FPI in three blocks")
    for i in range(org_model.num_hidden_layers):
        temp_layer = modulelist.__getitem__(i)
        set_bert_layer_fpi(new_model, org_model, temp_layer, org_hidden_size, target_hidden_size, level=i)

    encoder_layers = new_encoder.name_cells()
    # Get the ModuleList of layers
    layers = list(encoder_layers.values())
    modulelist = layers[0]

    # Step 2 FFN extension and MHA extension (if necessary)
    logger.critical(f"Step 2 Start: Use the {method} policy to extend FFN or MHA (if necessary)")
    # FFN
    if new_model.intermediate_size > org_model.intermediate_size:
        logger.critical("FFN extension start")
        dic = dict()
        if method == "FPI":
            for i in range(org_model.num_hidden_layers):
                temp_layer = modulelist.__getitem__(i)
                ffn_block = find_ffn_block(temp_layer)
                temp_dic = set_ffn_fpi(new_model, ffn_block, org_intermediate_size=org_model.intermediate_size,
                                       target_intermediate_size=new_model.intermediate_size)
                dic.update(temp_dic)
        elif method == "AKI":
            temp_layer = modulelist.__getitem__(0)
            for i in range(org_model.num_hidden_layers - 1):
                nxt_layer = modulelist.__getitem__(i + 1)
                ffn_block = find_ffn_block(temp_layer)
                ffn_block_nxt = find_ffn_block(nxt_layer)
                temp_dict = set_ffn_aki(new_model, ffn_block, ffn_block_nxt, org_model.intermediate_size,
                                        new_model.intermediate_size)
                temp_layer = nxt_layer
                dic.update(temp_dict)
            ffn_block = find_ffn_block(temp_layer)
            temp_dict = set_ffn_fpi(new_model, ffn_block, org_model.intermediate_size, new_model.intermediate_size)
            dic.update(temp_dict)

        else:
            raise Exception
        load_param_into_net(new_model, dic, strict_load=False)

    # Multihead extension
    if new_model.num_attention_heads > org_model.num_attention_heads:
        logger.critical("MHA expansion start")
        dic = dict()
        if method == "FPI":
            for i in range(org_model.num_hidden_layers):
                temp_layer = modulelist.__getitem__(i)
                mha_block = find_mha_block(temp_layer)
                temp_dic = set_mha_fpi(new_model, mha_block, org_head_num=org_model.num_attention_heads,
                                       target_head_num=new_model.num_attention_heads)
                dic.update(temp_dic)
        elif method == "AKI":
            temp_layer = modulelist.__getitem__(0)
            for i in range(org_model.num_hidden_layers - 1):
                nxt_layer = modulelist.__getitem__(i + 1)
                mha_block = find_mha_block(temp_layer)
                mha_block_nxt = find_mha_block(nxt_layer)
                temp_dict = set_mha_aki(new_model, mha_block, mha_block_nxt, org_head_num=org_model.num_attention_heads,
                                        target_head_num=new_model.num_attention_heads)
                temp_layer = nxt_layer
                dic.update(temp_dict)
            mha_block = find_mha_block(temp_layer)
            temp_dict = set_mha_fpi(new_model, mha_block, org_head_num=org_model.num_attention_heads,
                                    target_head_num=new_model.num_attention_heads)
            dic.update(temp_dict)

        else:
            raise Exception
        load_param_into_net(new_model, dic, strict_load=False)

    # Deep expansion. Algorithm 1
    org_num_layers = org_model.num_hidden_layers
    if new_num_layers is None or org_num_layers == new_num_layers:
        pass
    else:
        # Is the calculation divisible? Cannot divide, then n= 0
        k, n = new_num_layers // org_num_layers, new_num_layers % org_num_layers
        logger.critical(f"Depth expansion start: copy {k - 1} times vertically, and complement {n} bits with high bits")
        # Find the modified encoder in the new model_ block
        paras_dict = new_model.parameters_dict()

        # Convert Encoder to org_ num_ Bertlayer extraction of layers to form encoder_ block
        encoder_block = dict()
        for layer_name in paras_dict.keys():
            if "encoder" in layer_name:
                if int(find_number(layer_name)) < org_num_layers:
                    # Composition encoder_ block
                    encoder_block[layer_name] = paras_dict.get(layer_name)

        depth_dict = dict()
        for i in range(1, k):
            start = i * org_num_layers
            end = start + org_num_layers
            tmp_dict = set_depth(new_model, encoder_block, org_num_layers, start, end)
            depth_dict.update(tmp_dict)
        # Surplus n layers shall be filled with higher layers
        tmp_dict = set_depth(new_model, encoder_block, org_num_layers, k * org_num_layers, k * org_num_layers + n)
        depth_dict.update(tmp_dict)
        load_param_into_net(new_model, depth_dict, strict_load=False)

    # Expand the parameters of the classifier
    logger.critical("Start extending classifier parameters")
    dense_dict = set_dense(new_model, org_model, org_hidden_size, target_hidden_size)
    load_param_into_net(new_model, dense_dict, strict_load=False)


def set_depth(new_model, encoder_block, num_layers, start_idx, end_idx):
    """
    Stack encoder blocks in the depth direction
    :param new_model: new model
    :param encoder_block: org has completed AKI or FPI_ num_ The encoder block of layers is a state_ dict
    :param num_layers: encoder_ Number of blocks
    :param start_idx: subscript of layer to be stacked
    :param end_idx: subscript of tail layer
    """
    temp_dict = dict()
    for idx in range(start_idx, end_idx):
        # Equal is the lower layer corresponding to idx
        # Complementary part, use the parameter relative to top
        if end_idx - start_idx != num_layers:
            equal = idx % num_layers + num_layers - (end_idx - start_idx)
        else:
            equal = idx % num_layers
        # Copy the parameters of the low layer layer, change the name,
        # and put it into the current dict
        for name in encoder_block.keys():
            num_lay = find_number(name)
            if str(equal) == num_lay:
                temp_name = name.replace(str(equal), str(idx), 1)
                logger.info(f"{name}-->{temp_name}")
                layer = encoder_block.get(name)
                para = mindspore.Parameter(layer.data, name=temp_name)
                temp_dict[temp_name] = para
    return temp_dict


def set_bert_layer_fpi(new_model, org_model, bert_layer, org_hidden_size, target_hidden_size, level):
    """
    bert_ Layer: defined BertLayer structure
    :param new_model: new model
    :param org_model: original model
    :param bert_layer: 1 encoder to be operated
    :param org_hidden_size: initial hidden size
    :param target_hidden_size: target hidden size
    :param level: encoder series
    """
    all_layers = list(bert_layer.get_parameters())
    block1 = []
    block2 = []
    block3 = []
    # In the first level encoder, you need to add embedding_ table
    for param in all_layers:
        name = param.name
        if "query" in name or "key" in name or "value" in name:
            block1.append(param)
        elif "attention.output" in name or "intermediate" in name:
            block2.append(param)
        elif "output" in name:
            block3.append(param)

    if level == 0:
        embeddings = find_embeddings(org_model)
        block1 = embeddings + block1
    dic1 = set_block1(new_model, block1, org_hidden_size, target_hidden_size)
    dic2 = set_block2(new_model, block2, org_hidden_size, target_hidden_size)
    dic3 = set_block3(new_model, block3, org_hidden_size, target_hidden_size)
    dic1.update(dic2)
    dic1.update(dic3)
    load_param_into_net(new_model, dic1, strict_load=False)


def set_dense(new_model, org_model, org_hidden_size, target_hidden_size):
    """
    Extended classifier parameters
    :param new_model: new model
    :param org_model: W_ l2 W_ LN
    :param org_hidden_size: initial hidden layer size
    :param target_hidden_size: target hidden layer size
    """
    dense_block = find_dense_weight(org_model)
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    for param in dense_block:
        key = param.name
        weights = param
        if "weight" in key:
            # Expand input first
            weights1 = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                  target_col=target_hidden_size, to_expand="col")
            dic[key] = expand_copy(weights1, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                   target_col=target_hidden_size, to_expand="row")
        elif "bias" in key:
            dic[key] = expand_copy(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                   target_col=1, to_expand="row")
        else:
            raise Exception

        logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")
    for key in dic.keys():
        dic[key] = mindspore.Parameter(dic.get(key), name=key)
    return dic
