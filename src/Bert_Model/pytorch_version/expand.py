import random

import torch
from loguru import logger


def random_choice(x):
    """
    Randomly select a number in [0, x)
    :param x: upper bound
    """
    return random.randint(0, x - 1)


def generate_random_match(org_size, target_size):
    """
    Generate Random Matching Dictionary.[org_size, target_size) --> [0,org_size)
    :param org_size: origin size
    :param target_size: target size
    """
    match_dict = dict()
    for i in range(org_size, target_size):
        match_dict[i] = random_choice(org_size)
    return match_dict


def expand_fpi(org_matrix, target_row, target_col, choose_num_dict, to_expand):
    """
    Expand the parameter matrix according to the FPI policy. The input format of the matrix is (d_out, d_in)
    :param org_matrix: Original parameter matrix
    :param target_row:
    :param target_col:
    :param choose_num_dict: Match Selection Dictionary
    :param to_expand: Extension strategy
    :return: Extended parameter matrix
    """
    flag = 0
    if org_matrix.ndim == 1:
        flag = 1
        org_matrix = org_matrix.view(-1, 1)
    logger.info(f"Performing FPI extension {to_expand}")
    assert org_matrix.ndim == 2
    row, col = org_matrix.shape
    if target_row < row or target_col < col:
        raise Exception("expanded row or col smaller than origin")

    new = torch.zeros((target_row, target_col), dtype=torch.float32)
    new[:row, :col] = org_matrix[:, :]

    # Expand columns first, in dimension expansion
    # First select the corresponding g (i) for each column to be expanded, and then calculate C (g (i))
    # Used for counting
    count = dict()
    for choice in choose_num_dict.values():
        count[choice] = count.get(choice, 0) + 1

    if to_expand == "col":
        # divide the corresponding row of the original matrix by C (g (i)) Formula (7)
        for to_divide_col in count.keys():
            new[:row, to_divide_col] /= count.get(to_divide_col) + 1

        col = min(choose_num_dict.keys())
        for temp_col in range(col, target_col):
            choice = choose_num_dict.get(temp_col)
            new[:row, temp_col] = new[:row, choice]

    if to_expand == "row":
        # Expand row, out limitation expansion formula (8)

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
    Follow direct_ The copy policy extends the parameter matrix. The input format of the matrix is (d_out, d_in)
    :param org_matrix: Original parameter matrix
    :param target_row:
    :param target_col:
    :param choose_num_dict: Match Selection Dictionary
    :param to_expand: Extension strategy
    :return: Extended parameter matrix
    """
    flag = 0
    if org_matrix.ndim == 1:
        flag = 1
        org_matrix = org_matrix.view(-1, 1)
    assert org_matrix.ndim == 2
    row, col = org_matrix.shape

    if target_row < row or target_col < col:
        raise Exception("expanded row or col smaller than origin")

    new = torch.zeros((target_row, target_col), dtype=torch.float32)
    new[:row, :col] = org_matrix[:row, :col]

    # Extension column, in dimension expansion
    # First select the corresponding g (i) for each column to be expanded, and then calculate C (g (i))
    # Used for counting
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

    if flag == 1:
        new = new.view(-1)
    logger.info(f"Performing Copy expansion {to_expand}")
    return new


def expand_aki(org_matrix, nxt_matrix, target_row, target_col, choose_num_dict, to_expand):
    """
    Advanced knowledge expansion, only for FFN and MHA
    :param org_matrix: parameter matrix of the current layer
    :param nxt_matrix: parameter matrix of the next layer
    :param target_row: number of target rows
    :param target_col: Number of target columns
    :param choose_num_dict: matching selection dictionary
    :param to_expand: extension policy
    : return: Extended parameter matrix
    """
    assert org_matrix.ndim == nxt_matrix.ndim
    flag = 0
    if org_matrix.ndim == 1:
        org_matrix = org_matrix.view(-1, 1)
        nxt_matrix = nxt_matrix.view(-1, 1)
        flag = 1
    row, col = org_matrix.shape
    logger.info(f"Executing AKI extension {to_expand}")

    if target_row < row or target_col < col:
        raise Exception("expanded row or col smaller than origin")

    new_output = torch.zeros((target_row, target_col), dtype=torch.float32)
    new_output[:row, :col] = org_matrix[:, :]

    # Next level matrix
    new_nxt = torch.zeros((row, target_col), dtype=torch.float32)
    new_nxt[:row, :col] = nxt_matrix[:, :]

    count = dict()
    for choice in choose_num_dict.values():
        count[choice] = count.get(choice, 0) + 1

    if to_expand == "col":
        # ① Extended temp_ Columns of matrix
        # First select the corresponding g (i) for each column to be expanded, and then calculate C (g (i))
        # According to the value of C (g (i)),
        # divide the corresponding row of the original matrix by C (g (i)) Formula (7)
        for to_divide_col in count.keys():
            # Directly operate the next layer parameters
            new_nxt[:row, to_divide_col] /= count.get(to_divide_col) + 1
            new_output[:row, to_divide_col] /= count.get(to_divide_col) + 1

        col = min(choose_num_dict.keys())
        for temp_col in range(col, target_col):
            choice = choose_num_dict.get(temp_col)
            # Select from the next layer
            new_output[:row, temp_col] = new_nxt[:row, choice]
    if to_expand == "row":
        # Extended temp_ Row of matrix
        for to_divide_row in count.keys():
            # Directly operate the next layer parameters
            new_nxt[to_divide_row, :col] /= count.get(to_divide_row) + 1
            new_output[to_divide_row, :col] /= count.get(to_divide_row) + 1

        row = min(choose_num_dict.keys())
        for temp_row in range(row, target_row):
            choice = choose_num_dict.get(temp_row)
            new_output[temp_row, :col] = new_nxt[choice, :col]
    if flag:
        new_output = new_output.view(-1)
    return new_output


def expand_aki_copy(org_matrix, nxt_matrix, target_row, target_col, choose_num_dict, to_expand):
    """
    Advanced knowledge expansion, only for FFN and MHA
    :param org_matrix: parameter matrix of the current layer
    :param nxt_matrix: parameter matrix of the next layer
    :param target_row: number of target rows
    :param target_col: Number of target columns
    :param choose_num_dict: matching selection dictionary
    :param to_expand: extension policy
    : return: Extended parameter matrix
    """
    assert org_matrix.ndim == nxt_matrix.ndim
    flag = 0
    if org_matrix.ndim == 1:
        org_matrix = org_matrix.view(-1, 1)
        nxt_matrix = nxt_matrix.view(-1, 1)
        flag = 1
    row, col = org_matrix.shape
    logger.info(f"Executing AKI_ Copy extension {to_expand}")

    if target_row < row or target_col < col:
        raise Exception("expanded row or col smaller than origin")

    new_output = torch.zeros((target_row, target_col), dtype=torch.float32)
    new_output[:row, :col] = org_matrix[:, :]
    # Next level matrix
    new_nxt = torch.zeros((row, target_col), dtype=torch.float32)
    new_nxt[:row, :col] = nxt_matrix[:, :]
    if to_expand == "col":
        # ① Extended temp_ Columns of matrix
        # First select the corresponding g (i) for each column to be expanded, and then calculate C (g (i))
        # According to the value of C (g (i)),
        # divide the corresponding row of the original matrix by C (g (i)) Formula (7)
        col = min(choose_num_dict.keys())
        for temp_col in range(col, target_col):
            choice = choose_num_dict.get(temp_col)
            new_output[:row, temp_col] = new_nxt[:row, choice]
    if to_expand == "row":
        row = min(choose_num_dict.keys())
        for temp_row in range(row, target_row):
            choice = choose_num_dict.get(temp_row)
            new_output[temp_row, :col] = new_nxt[choice, :col]
    if flag:
        new_output = new_output.view(-1)
    return new_output
