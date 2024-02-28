import torch.nn
from loguru import logger

from .expand import expand_fpi, expand_aki, expand_copy, expand_aki_copy, \
    generate_random_match
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
    head_size = int(new_model.config.size_per_head * new_model.config.num_attention_heads)

    for (key, param) in org_block:
        weights = param
        if "embeddings" in key:
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
                dic[key] = torch.concat([weights, torch.zeros(target_hidden_size - weights.shape[0])], 0)
            else:
                raise Exception
        else:
            raise Exception("This layer does not exist in block 1")

        if key in dic.keys():
            logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")

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
    head_size = int(new_model.config.num_attention_heads * new_model.config.size_per_head)
    intermediate = new_model.config.intermediate_size
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    for key, param in org_block:
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

        elif "intermediate" in key:
            if "weight" in key:
                dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=intermediate,
                                      target_col=target_hidden_size, to_expand="col")
            elif "bias" in key:
                # Expand the bios in the FFN extension part. No operation is required here, just copy
                dic[key] = torch.concat([weights, torch.zeros(intermediate - weights.shape[0])], 0)
            else:
                raise Exception
        else:
            raise Exception(f"{key} layer does not exist in block 2")
        if key in dic.keys():
            logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")
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
    intermediate = new_model.config.intermediate_size
    choose_num_dict = generate_random_match(org_hidden_size, target_hidden_size)
    dic = dict()
    for key, param in org_block:
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
            elif "LayerNorm" in key:
                if weights.ndim == 1:
                    dic[key] = expand_fpi(weights, choose_num_dict=choose_num_dict, target_row=target_hidden_size,
                                          target_col=1, to_expand="row")
                else:
                    raise Exception
        else:
            raise Exception(f"{key} layer does not exist in block 3")
        logger.info(f"{key}:  {weights.shape}  -->  {dic[key].shape}")
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
    hidden_size = new_model.config.hidden_size
    dic = dict()
    logger.debug(ffn_block)
    for key, param in ffn_block:
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
                dic[key] = torch.concat([weights, torch.zeros(hidden_size - weights.shape[0])], 0)

            else:
                raise Exception
        else:
            raise Exception
        logger.info(f"FFN_FPI: expand {key}: ")
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
    hidden_size = new_model.config.hidden_size
    dic = dict()
    for (key, param), (key_nxt, param_nxt) in zip(ffn_block, ffn_block_nxt):
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
                dic[key] = torch.concat([weights, torch.zeros(hidden_size - weights.shape[0])], 0)
            else:
                raise Exception
        else:
            raise Exception
        if key in dic.keys():
            logger.info(f"FFN_AKI: use {key_nxt} expand {key}: ")
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
    size_per_head = int(new_model.config.size_per_head)
    num_heads = new_model.config.num_attention_heads
    head_size = int(size_per_head * num_heads)
    choose_head_dict = generate_random_match(org_head_num, target_head_num)
    new_dict = dict()
    # Extend the set to fit the input requirements
    for key in choose_head_dict.keys():
        chosen_span = [*range(choose_head_dict.get(key) * size_per_head,
                              choose_head_dict.get(key) * size_per_head + size_per_head)]
        choose_span = [*range(key * size_per_head, key * size_per_head + size_per_head)]
        for after, pre in zip(choose_span, chosen_span):
            new_dict[after] = pre
    dic = dict()
    for key, param in mha_block:
        weights = param
        if "query" in key or "key" in key or "value" in key:
            # Extended output, only copy is required
            if "weight" in key:
                dic[key] = expand_copy(weights, target_row=head_size, target_col=new_model.config.hidden_size,
                                       choose_num_dict=new_dict, to_expand="row")
            elif "bias" in key:
                dic[key] = expand_copy(weights, target_row=head_size, target_col=1,
                                       choose_num_dict=new_dict, to_expand="row")
            else:
                raise Exception
        elif "output" in key:
            # Extended output, fpi is required
            if "weight" in key:
                dic[key] = expand_fpi(weights, target_row=new_model.config.hidden_size, target_col=head_size,
                                      choose_num_dict=new_dict, to_expand="col")
            elif "bias" in key:
                dic[key] = torch.concat([weights, torch.zeros(new_model.config.hidden_size - weights.shape[0])], 0)
            else:
                raise Exception
        if key in dic.keys():
            logger.info(f"MHA_FPI: expand {key}: ")
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
    size_per_head = int(new_model.config.size_per_head)
    num_heads = new_model.config.num_attention_heads
    head_size = int(size_per_head * num_heads)
    choose_head_dict = generate_random_match(org_head_num, target_head_num)
    new_dict = dict()
    for key in choose_head_dict.keys():
        chosen_span = [*range(choose_head_dict.get(key) * size_per_head,
                              choose_head_dict.get(key) * size_per_head + size_per_head)]
        choose_span = [*range(key * size_per_head, key * size_per_head + size_per_head)]
        for after, pre in zip(choose_span, chosen_span):
            new_dict[after] = pre
    dic = dict()
    for (key, param), (key_nxt, param_nxt) in zip(mha_block, mha_block_nxt):
        weights = param
        weights_nxt = param_nxt
        if "query" in key or "key" in key or "value" in key:
            if "weight" in key:
                dic[key] = expand_aki_copy(weights, weights_nxt, target_row=head_size,
                                           target_col=new_model.config.hidden_size,
                                           choose_num_dict=new_dict, to_expand="row")
            elif "bias" in key:
                dic[key] = expand_aki_copy(weights, weights_nxt, target_row=head_size, target_col=1,
                                           choose_num_dict=new_dict, to_expand="row")
            else:
                raise Exception
        elif "output" in key:
            if "weight" in key:
                dic[key] = expand_aki(weights, weights_nxt, target_row=new_model.config.hidden_size,
                                      target_col=head_size,
                                      choose_num_dict=new_dict, to_expand="col")
            elif "bias" in key:
                dic[key] = torch.concat([weights, torch.zeros(new_model.config.hidden_size - weights.shape[0])], 0)
            else:
                raise Exception
        if key in dic.keys():
            logger.info(f"MHA_AKI: use {key_nxt} expand {key}: ")
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
    encoder_layers = org_encoder.named_children()
    # Get the ModuleList of layers
    layers = list(encoder_layers)
    modulelist = layers[0][1]
    org_layers = org_model.config.num_hidden_layers

    # Get every BertEncoderCell in ModuleList

    # The last layer can only use FPI
    if modulelist.__len__() == 1:
        method = "FPI"

    # Step 1 Run the update step 1 for each layer.
    # Just reduce the depth of the model. Whether AKI or FPI
    logger.critical(f"Step 1 starts: {org_model.config.num_hidden_layers} encoders. FPI in three blocks")

    for i in range(org_model.config.num_hidden_layers):
        temp_layer = modulelist.__getitem__(i)
        set_bert_layer_fpi(new_model, org_model, temp_layer, org_hidden_size, target_hidden_size, level=i,
                           prefix=f"encoder.layer.{i}.")

    # After step 1, use the new matrix to complete the subsequent expansion
    encoder_layers = new_encoder.named_children()
    # Get the ModuleList of layers
    layers = list(encoder_layers)
    modulelist = layers[0][1]

    # Step 2 FFN extension and MHA extension (if necessary)
    logger.critical(f"Step 2 Start: Use the {method} policy to extend FFN or MHA (if necessary)")

    # FFN
    if new_model.config.intermediate_size > org_model.config.intermediate_size:
        logger.critical("FFN expansion start")
        dic = dict()
        if method == "FPI":
            for i in range(org_layers):
                temp_layer = modulelist.__getitem__(i)
                ffn_block = find_ffn_block(temp_layer, f"encoder.layer.{i}.")
                temp_dic = set_ffn_fpi(new_model, ffn_block, org_intermediate_size=org_model.config.intermediate_size,
                                       target_intermediate_size=new_model.config.intermediate_size)
                dic.update(temp_dic)
        elif method == "AKI":
            temp_layer = modulelist.__getitem__(0)
            for i in range(org_layers - 1):
                nxt_layer = modulelist.__getitem__(i + 1)
                ffn_block = find_ffn_block(temp_layer, f"encoder.layer.{i}.")
                ffn_block_nxt = find_ffn_block(nxt_layer, f"encoder.layer.{i + 1}.")
                temp_dict = set_ffn_aki(new_model, ffn_block, ffn_block_nxt, org_model.config.intermediate_size,
                                        new_model.config.intermediate_size)
                temp_layer = nxt_layer
                dic.update(temp_dict)
            ffn_block = find_ffn_block(temp_layer, f"encoder.layer.{org_layers - 1}.")
            temp_dict = set_ffn_fpi(new_model, ffn_block, org_model.config.intermediate_size,
                                    new_model.config.intermediate_size)
            dic.update(temp_dict)

        else:
            raise Exception
        logger.info("Import Parameters")
        logger.info(dic.keys())
        new_model.load_state_dict(dic, strict=False)

    # Multihead extension
    if new_model.config.num_attention_heads > org_model.config.num_attention_heads:
        logger.critical("MHA expansion start")
        dic = dict()
        if method == "FPI":
            for i in range(org_layers):
                temp_layer = modulelist.__getitem__(i)
                mha_block = find_mha_block(temp_layer, f"encoder.layer.{i}.")
                temp_dic = set_mha_fpi(new_model, mha_block, org_head_num=org_model.config.num_attention_heads,
                                       target_head_num=new_model.config.num_attention_heads)
                dic.update(temp_dic)
        elif method == "AKI":
            temp_layer = modulelist.__getitem__(0)
            for i in range(org_layers - 1):
                nxt_layer = modulelist.__getitem__(i + 1)
                mha_block = find_mha_block(temp_layer, f"encoder.layer.{i}.")
                mha_block_nxt = find_mha_block(nxt_layer, f"encoder.layer.{i + 1}.")
                temp_dict = set_mha_aki(new_model, mha_block, mha_block_nxt,
                                        org_head_num=org_model.config.num_attention_heads,
                                        target_head_num=new_model.config.num_attention_heads)
                temp_layer = nxt_layer
                dic.update(temp_dict)
            mha_block = find_mha_block(temp_layer, f"encoder.layer.{org_layers - 1}.")
            temp_dict = set_mha_fpi(new_model, mha_block, org_head_num=org_model.config.num_attention_heads,
                                    target_head_num=new_model.config.num_attention_heads)
            dic.update(temp_dict)

        else:
            raise Exception
        logger.info("Import Parameters")
        logger.info(dic.keys())
        new_model.load_state_dict(dic, strict=False)

    # Deep expansion. Algorithm 1
    if new_num_layers is None or org_layers == new_num_layers:
        pass
    else:
        # Is the calculation divisible? Cannot divide, then n= 0
        k, n = new_num_layers // org_layers, new_num_layers % org_layers
        logger.critical(f"Depth expansion start: copy {k - 1} times vertically, and complement {n} bits with high bits")
        # Find the modified encoder in the new model_ block

        paras_dict = new_model.state_dict()
        # Convert Encoder to org_ num_ Bertlayer extraction of layers to form encoder_ block

        encoder_block = dict()
        for layer_name in paras_dict.keys():
            if "encoder" in layer_name:
                if int(find_number(layer_name)) < org_layers:
                    # 组成encoder_block
                    encoder_block[layer_name] = paras_dict.get(layer_name)

        for i in range(1, k):
            start = i * org_layers
            end = start + org_layers
            set_depth(new_model, encoder_block, org_layers, start, end)
        # Surplus n layers shall be filled with higher layers
        set_depth(new_model, encoder_block, org_layers, k * org_layers, k * org_layers + n)
    # Expand the parameters of the classifier
    logger.critical("Start extending classifier parameters")
    dense_dict = set_dense(new_model, org_model, org_hidden_size, target_hidden_size)
    new_model.load_state_dict(dense_dict, strict=False)


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
        # Copy the parameters of the low layer layer, change the name, and put it into the current dict
        for name in encoder_block.keys():
            num_lay = find_number(name)
            if str(equal) == num_lay:
                temp_name = name.replace(str(equal), str(idx), 1)
                logger.info(f"{name}-->{temp_name}")
                layer = encoder_block.get(name)
                temp_dict[temp_name] = layer.data
    new_model.load_state_dict(temp_dict, strict=False)


def set_bert_layer_fpi(new_model, org_model, bert_layer, org_hidden_size, target_hidden_size, level, prefix):
    """
    bert_ Layer: defined BertLayer structure
    :param new_model: new model
    :param org_model: original model
    :param bert_layer: 1 encoder to be operated
    :param org_hidden_size: initial hidden size
    :param target_hidden_size: target hidden size
    :param level: encoder series
    """
    all_layers = list(bert_layer.named_parameters())
    block1 = []
    block2 = []
    block3 = []
    # In the first level encoder, you need to add embedding_ table
    for name, param in all_layers:
        name = prefix + name
        if "query" in name or "key" in name or "value" in name:
            block1.append((name, param))
        elif "attention.output" in name or "intermediate" in name:
            block2.append((name, param))
        elif "output" in name:
            block3.append((name, param))

    if level == 0:
        embeddings = find_embeddings(org_model)
        block1 = embeddings + block1
    dic1 = set_block1(new_model, block1, org_hidden_size, target_hidden_size)
    dic2 = set_block2(new_model, block2, org_hidden_size, target_hidden_size)
    dic3 = set_block3(new_model, block3, org_hidden_size, target_hidden_size)
    dic1.update(dic2)
    dic1.update(dic3)
    logger.critical("Import Parameters")
    logger.info(dic1.keys())
    warn = new_model.load_state_dict(dic1, strict=False)


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
    for key, weights in dense_block:
        if "weight" in key:
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
    return dic
