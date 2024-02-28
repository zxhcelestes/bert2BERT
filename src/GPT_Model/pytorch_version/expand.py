import random

import torch
from loguru import logger


def random_choice(x):
    """
    在[0,x)随机选取一个数
    :param x: 上界
    """
    return random.randint(0, x - 1)


def generate_random_match(org_size, target_size):
    """
    生成随机匹配字典。从[org_size, target_size) --> [0,org_size)
    :param org_size: 初始规模
    :param target_size: 目标规模
    """
    match_dict = dict()
    for i in range(org_size, target_size):
        match_dict[i] = random_choice(org_size)
    return match_dict


def expand_fpi(org_matrix, target_row, target_col, choose_num_dict, to_expand):
    """
    按照FPI策略扩展参数矩阵，矩阵输入格式为(d_out,d_in)
    :param org_matrix: 原参数矩阵
    :param target_row: 目标行数
    :param target_col: 目标列数
    :param choose_num_dict: 匹配选择字典
    :param to_expand: 扩展策略
    :return: 扩展参数矩阵
    """
    flag = 0
    if org_matrix.ndim == 1:
        flag = 1
        org_matrix = org_matrix.view(-1, 1)
    logger.info(f"正在执行FPI扩展{to_expand}")
    assert org_matrix.ndim == 2
    row, col = org_matrix.shape
    if target_row < row or target_col < col:
        logger.debug(f"org_row:{row} , row: {target_row} , org_col{col} ,col {target_col}")
        raise Exception("expanded row or col smaller than origin")

    new = torch.zeros((target_row, target_col), dtype=torch.float32)
    new[:row, :col] = org_matrix[:, :]

    # 先扩展列，in-dimension-expansion
    # 先为待扩展的每一列选定其对应的g(i),然后统计得到C(g(i))
    # 用于计数
    count = dict()
    # 用于映射
    for choice in choose_num_dict.values():
        count[choice] = count.get(choice, 0) + 1

    if to_expand == "col":
        # 根据C(g(i))的值，对原矩阵的相应行除以C(g(i)) 公式(7)
        for to_divide_col in count.keys():
            new[:row, to_divide_col] /= count.get(to_divide_col) + 1

        col = min(choose_num_dict.keys())
        for temp_col in range(col, target_col):
            choice = choose_num_dict.get(temp_col)
            new[:row, temp_col] = new[:row, choice]

    if to_expand == "row":
        # 扩展row，out-dimension-expansion 公式(8)
        for to_divide_row in count.keys():
            new[to_divide_row, :col] /= count.get(to_divide_row) + 1
        row = min(choose_num_dict.keys())
        for temp_row in range(row, target_row):
            choice = choose_num_dict.get(temp_row)
            new[temp_row, :] = new[choice, :]

    if flag == 1:
        new = new.view(-1)
    return new


def expand_copy(org_matrix, target_row, target_col, choose_num_dict, to_expand):
    """
    按照direct_copy策略扩展参数矩阵，矩阵输入格式为(d_out,d_in)
    :param org_matrix: 原参数矩阵
    :param target_row: 目标行数
    :param target_col: 目标列数
    :param choose_num_dict: 匹配选择字典
    :param to_expand: 扩展策略
    :return: 扩展参数矩阵
    """
    flag = 0
    if org_matrix.ndim == 1:
        flag = 1
        org_matrix = org_matrix.view(-1, 1)
    assert org_matrix.ndim == 2
    row, col = org_matrix.shape

    if target_row < row or target_col < col:
        logger.debug(f"org_row:{row} , row: {target_row} , org_col{col} ,col {target_col}")
        raise Exception("expanded row or col smaller than origin")

    new = torch.zeros((target_row, target_col), dtype=torch.float32)
    new[:row, :col] = org_matrix[:row, :col]

    # 扩展列，in-dimension-expansion
    # 先为待扩展的每一列选定其对应的g(i),然后统计得到C(g(i))
    if to_expand == "col":
        col = min(choose_num_dict.keys())
        for temp_col in range(col, target_col):
            choice = choose_num_dict.get(temp_col)
            new[:row, temp_col] = new[:row, choice]

    if to_expand == "row":
        row = min(choose_num_dict.keys())
        for temp_row in range(row, target_row):
            choice = choose_num_dict.get(temp_row)
            new[temp_row, :] = new[choice, :]
    logger.debug(new.shape)
    if flag == 1:
        new = new.view(-1)
    logger.info(f"正在执行Copy扩展{to_expand}")
    return new


def expand_aki(org_matrix, nxt_matrix, target_row, target_col, choose_num_dict, to_expand):
    """
    超前知识扩展,仅用于FFN和MHA
    :param org_matrix: 当前层的参数矩阵
    :param nxt_matrix: 下一层的参数矩阵
    :param target_row: 目标行数
    :param target_col: 目标列数
    :param choose_num_dict: 匹配选择字典
    :param to_expand: 扩展策略
    :return: 扩展后的参数矩阵
    """
    assert org_matrix.ndim == nxt_matrix.ndim
    flag = 0
    if org_matrix.ndim == 1:
        org_matrix = org_matrix.view(-1, 1)
        nxt_matrix = nxt_matrix.view(-1, 1)
        flag = 1
    row, col = org_matrix.shape
    logger.info(f"正在执行AKI扩展{to_expand}")

    if target_row < row or target_col < col:
        raise Exception("expanded row or col smaller than origin")

    new_output = torch.zeros((target_row, target_col), dtype=torch.float32)
    new_output[:row, :col] = org_matrix[:, :]

    # 下一层的矩阵
    new_nxt = torch.zeros((row, target_col), dtype=torch.float32)
    new_nxt[:row, :col] = nxt_matrix[:, :]

    count = dict()
    # 用于映射
    for choice in choose_num_dict.values():
        count[choice] = count.get(choice, 0) + 1

    if to_expand == "col":
        # ①扩展temp_matrix的列
        # 先为待扩展的每一列选定其对应的g(i),然后统计得到C(g(i))
        # 根据C(g(i))的值，对原矩阵的相应行除以C(g(i)) 公式(7)
        for to_divide_col in count.keys():
            # 直接操作下一层参数
            new_nxt[:row, to_divide_col] /= count.get(to_divide_col) + 1
            new_output[:row, to_divide_col] /= count.get(to_divide_col) + 1

        col = min(choose_num_dict.keys())
        for temp_col in range(col, target_col):
            choice = choose_num_dict.get(temp_col)
            # 从下一层选
            new_output[:row, temp_col] = new_nxt[:row, choice]
    if to_expand == "row":
        # 扩展temp_matrix的行
        for to_divide_row in count.keys():
            # 直接操作下一层参数
            new_nxt[to_divide_row, :col] /= count.get(to_divide_row) + 1
            new_output[to_divide_row, :col] /= count.get(to_divide_row) + 1

        row = min(choose_num_dict.keys())
        for temp_row in range(row, target_row):
            choice = choose_num_dict.get(temp_row)
            # 从下一层选
            new_output[temp_row, :col] = new_nxt[choice, :col]
    if flag:
        new_output = new_output.view(-1)
    return new_output


def expand_aki_copy(org_matrix, nxt_matrix, target_row, target_col, choose_num_dict, to_expand):
    """
    超前知识扩展,仅用于FFN和MHA
    :param org_matrix: 当前层的参数矩阵
    :param nxt_matrix: 下一层的参数矩阵
    :param target_row: 目标行数
    :param target_col: 目标列数
    :param choose_num_dict: 匹配选择字典
    :param to_expand: 扩展策略
    :return: 扩展后的参数矩阵
    """
    assert org_matrix.ndim == nxt_matrix.ndim
    flag = 0
    if org_matrix.ndim == 1:
        org_matrix = org_matrix.view(-1, 1)
        nxt_matrix = nxt_matrix.view(-1, 1)
        flag = 1
    row, col = org_matrix.shape
    logger.info(f"正在执行AKI_copy扩展{to_expand}")

    if target_row < row or target_col < col:
        raise Exception("expanded row or col smaller than origin")

    new_output = torch.zeros((target_row, target_col), dtype=torch.float32)
    new_output[:row, :col] = org_matrix[:, :]
    # 下一层的矩阵
    new_nxt = torch.zeros((row, target_col), dtype=torch.float32)
    new_nxt[:row, :col] = nxt_matrix[:, :]
    if to_expand == "col":
        # ①扩展temp_matrix的列
        # 先为待扩展的每一列选定其对应的g(i),然后统计得到C(g(i))
        # 根据C(g(i))的值，对原矩阵的相应行除以C(g(i)) 公式(7)
        col = min(choose_num_dict.keys())
        for temp_col in range(col, target_col):
            choice = choose_num_dict.get(temp_col)
            # 从下一层选
            new_output[:row, temp_col] = new_nxt[:row, choice]
    if to_expand == "row":
        # 扩展temp_matrix的行
        row = min(choose_num_dict.keys())
        for temp_row in range(row, target_row):
            choice = choose_num_dict.get(temp_row)
            # 从下一层选
            new_output[temp_row, :col] = new_nxt[choice, :col]
    if flag:
        new_output = new_output.view(-1)
    return new_output
