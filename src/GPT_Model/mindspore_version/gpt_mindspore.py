# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
GPT-2 base model
"""
import copy
import math

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
import numpy as np
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P

from .weight_init import normal_weight, zero_weight


class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 use_one_hot_embeddings=False,
                 compute_type=mstype.float16):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.compute_type = compute_type
        self.embedding_table = Parameter(normal_weight([vocab_size, embedding_dim], embedding_dim),
                                         name='embedding_table')
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

    def construct(self, input_ids):
        """
        get embedding according to input_ids.

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.

        Returns:
            output (Tensor): the embedding matrix according to the input_ids.
            self.embedding_table (Parameter): the whole embedding table of GPT-2 model.
        """
        input_shape = self.shape(input_ids)  # [batch_size, seq_length]
        flat_ids = self.reshape(input_ids, self.shape_flat)  # [batch_size * seq_length]

        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)

            # precision transition fp32 -> fp16
            one_hot_ids = self.cast(one_hot_ids, self.compute_type)
            self.embedding_table = self.cast(self.embedding_table, self.compute_type)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
            output_for_reshape = self.cast(output_for_reshape, mstype.float32)

        else:
            # [batch_size * seq_length * embedding_dim]
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_dim,)
        output = self.reshape(output_for_reshape, out_shape)  # [batch_size, seq_length, embedidng_dim]
        return output, self.embedding_table.value()


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional embeddings to word embeddings.

    Args:
        embedding_dim (int): The size of each embedding vector.
        seq_length (int): the length of input sequence.
        n_positions (int): Maximum length of sequences used in this model. Default: 1024.
        dropout_prob (float): The dropout probability. Default: 0.1.
     """

    def __init__(self,
                 embedding_dim=None,
                 seq_length=None,
                 n_positions=1024,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()

        self.position_embedding_table = Parameter(
            normal_weight([n_positions, embedding_dim], embedding_dim), name='position_embeddings')
        self.expand_dims = P.ExpandDims()
        self.add = P.Add()
        self.gather = P.Gather()
        self.input_indices = Tensor(np.array([x for x in range(seq_length)]), mindspore.int32)
        self.dropout = nn.Dropout(1 - dropout_prob, dtype=mstype.float32)
        self.use_dropout = dropout_prob > 0

    def construct(self, word_embeddings):
        """
        Add the position embedding table to token embedding table
        Args:
            word_embeddings (Tensor): the token embedding matrix

        Returns:
            output (Tensor): the final embedding matrix by adding the position embedding table
                             to token embedding table.

        """
        position_embeddings = self.gather(self.position_embedding_table, self.input_indices, 0)
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(word_embeddings, position_embeddings)

        if self.use_dropout:
            output = self.dropout(output)
        return output


class CastWrapper(nn.Cell):
    """
    Cast wrapper
    """

    def __init__(self,
                 dst_type=mstype.float32):
        super(CastWrapper, self).__init__()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        """
        type cast
        Args:
            x (Tensor): the input which need to be cast.

        Returns:
            Tensor, the cast output.
        """
        return self.cast(x, self.dst_type)


class LayerNorm(nn.Cell):
    """
    Do layer norm

    Args:
        in_channels (int): In channels number of layer norm
    """

    def __init__(self,
                 in_channels=None):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm((in_channels,), epsilon=1e-5)
        self.cast = P.Cast()
        self.get_dtype = P.DType()

    def construct(self, input_tensor):
        """
        layer norm
        Args:
            input_tensor (Tensor): the input of layernorm.

        Returns:
            Tensor, the output after layernorm.
        """
        output = self.cast(input_tensor, mstype.float32)
        output = self.layer_norm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        return output


class ResidualConnection(nn.Cell):
    """
    Add residual to output.

    Args:
        dropout_prob (float): Dropout rate.
    """

    def __init__(self, dropout_prob=0.1):
        super(ResidualConnection, self).__init__()
        self.add = P.Add()
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.use_dropout = dropout_prob > 0

    def construct(self, hidden_tensor, input_tensor):
        """

        Args:
            hidden_tensor (Tensor): the output of sublayer.
            input_tensor (Tensor): the input tensor.

        Returns:
            output (Tensor): with the same shape of hidden_tensor.

        """
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        return output


class Conv1D(nn.Cell):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nx (int): The number of input features.
        nf (int): The number of output features.
    """

    def __init__(self,
                 nx,
                 nf):
        super(Conv1D, self).__init__()
        self.nx = nx
        self.nf = nf
        self.weight = Parameter(normal_weight([nx, nf], nf), name='projection_weight')
        self.bias = Parameter(zero_weight(nf), name='projection_bias')

        self.matmul = P.MatMul()
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()

    def construct(self, input_tensor):
        """

        Args:
            input_tensor (Tensor): the input tensor of Conv1D with shape [batch_size * seq_length, nx]

        Returns:
            output_tensor (Tensor): the output tensor with shape [batch_size * seq_length, self.nf]

        """
        # precision transition fp32 -> fp16
        input_tensor = self.cast(input_tensor, mstype.float16)
        fp16_weight = self.cast(self.weight, mstype.float16)
        output_tensor = self.matmul(input_tensor, fp16_weight)  # [batch_size * seq_length, self.nf]
        output_tensor = self.cast(output_tensor, mstype.float32)
        output_tensor = self.bias_add(output_tensor, self.bias)

        return output_tensor


class MaskedSelfAttention(nn.Cell):
    """
    Apply masked multi-head attention.

    Args:
        batch_size (int): Batch size of input datasets. Default: 512.
        n_embed (int): Size of last dim of input tensor. Default: 768.
        seq_length (int): Length of input tensor sequence. Default: 1024.
        n_head (int): Number of attention heads. Default: 12.
        dim_per_head (int): Size of each attention head. Default: 64.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: True.
        attention_dropout (float): The dropout probability for MultiheadAttention. Default: 0.0.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: mstype.float32.

    Returns:
        Tensor, with the shape [batch_size, seq_length, n_embed]

    """

    def __init__(self,
                 batch_size=512,
                 n_embed=768,
                 seq_length=1024,
                 n_head=12,
                 dim_per_head=64,
                 has_attention_mask=True,
                 do_return_2d_tensor=True,
                 attention_dropout=0.1,
                 compute_type=mstype.float16):
        super(MaskedSelfAttention, self).__init__()

        self.batch_size = batch_size
        self.n_embed = n_embed
        self.seq_length = seq_length
        self.num_heads = n_head
        self.dim_per_head = dim_per_head
        self.has_attention_mask = has_attention_mask
        self.compute_type = compute_type
        assert has_attention_mask

        self.scale = Tensor([1.0 / math.sqrt(float(self.dim_per_head))], dtype=compute_type)  # attention scale
        self.mask_data = Tensor([-10000.0, ], dtype=compute_type)
        self.split_head_shape = (-1, self.seq_length, self.num_heads, self.dim_per_head)

        self.c_attn = Conv1D(n_embed, n_embed * 3)
        self.c_proj = Conv1D(n_embed, n_embed)

        self.split_for_qkv = P.Split(1, 3)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.matmul = P.BatchMatMul()
        self.multiply = P.Mul()

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

        if do_return_2d_tensor:
            self.shape_return = (-1, n_embed)
        else:
            self.shape_return = (-1, seq_length, n_embed)

        self.softmax = nn.Softmax()
        self.softmax_cast = P.Cast()
        self.dropout = nn.Dropout(1 - attention_dropout)
        self.use_attention_dropout = attention_dropout > 0

    def construct(self, input_tensor, attention_mask):
        """
        do masked self-attention

        Args:
            input_tensor (Tensor): the embedding of input sequence tokens,
                                   shape with [batch_size * seq_length, d_mdoel]
            attention_mask (Tensor): mask to avoid performing attention on padding token indices,
                                     shape with [batch_size, seq_len, seq_len].

        Returns:
            outputs (Tensor): the output of masked self-attention, shape with [batch_size * seq_len, n_embed].
        """
        input_tensor = self.c_attn(input_tensor)  # [batch_size * seq_length, n_embed*3]---> eg.[1 * 3, 2304]
        input_tensor = self.split_for_qkv(input_tensor)
        query = input_tensor[0]  # [batch_size * seq_length, n_embed] ---> eg. [1 * 3, 768]
        key = input_tensor[1]
        value = input_tensor[2]

        # split head
        query = self.reshape(query, self.split_head_shape)
        # query shape [batch_size, num_heads, seq_len, dim_per_head] ---> eg. [1, 12, 3, 64]
        query = self.transpose(query, self.trans_shape)

        key = self.reshape(key, self.split_head_shape)
        # key shape [batch_size, num_heads, seq_len, dim_per_head] ---> eg. [1, 12, 3, 64]
        key = self.transpose(key, self.trans_shape)

        value = self.reshape(value, self.split_head_shape)
        # value shape [batch_size, num_heads, seq_len, dim_per_head] ---> eg. [1, 12, 3, 64]
        value = self.transpose(value, self.trans_shape)

        # attention and mask
        # precision transition fp32 -> fp16
        query = self.cast(query, self.compute_type)
        key = self.cast(key, self.compute_type)
        attention_scores = self.matmul_trans_b(query, key)  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = self.cast(attention_scores, self.compute_type)
        attention_scores = self.multiply(attention_scores, self.scale)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)  # [batch_size, 1, seq_length, seq_length]
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))  # fp16
            adder = self.multiply(multiply_out, self.mask_data)
            adder = self.cast(adder, mstype.float32)
            attention_scores = self.cast(attention_scores, mstype.float32)
            attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, mstype.float32)
        attention_probs = self.softmax(attention_scores)  # [batch_size, num_heads, seq_len, seq_len]
        attention_probs = self.softmax_cast(attention_probs, self.get_dtype(key))

        if self.use_attention_dropout:
            attention_probs = self.dropout(attention_probs)

        value = self.cast(value, mstype.float16)
        attention_probs = self.cast(attention_probs, self.compute_type)
        outputs = self.matmul(attention_probs, value)  # [batch_size, num_heads, seq_len, dim_per_head]
        outputs = self.cast(outputs, mstype.float32)

        # merge heads
        outputs = self.transpose(outputs, self.trans_shape)  # [batch_size, seq_len, num_heads, dim_per_head]
        outputs = self.reshape(outputs,
                               self.shape_return)  # default True, the outputs shape [batch_size * seq_len, n_embed]

        # project
        outputs = self.c_proj(outputs)

        return outputs


class FeedForward(nn.Cell):
    """
    Apply two-layer feed forward

    Args:
        in_channels (int): Size of the input layer. Default: 768.
        out_channels (int): Size of the output layers. Default: 768.
        hidden_size (int): Size of the hidden layer. Default: 3072.
        hidden_dropout (float): The dropout probability for hidden outputs. Default: 0.1.
    """

    def __init__(self,
                 in_channels=786,
                 out_channels=768,
                 hidden_size=3072,
                 hidden_dropout=0.1):
        super(FeedForward, self).__init__()

        self.c_fc = Conv1D(in_channels, hidden_size)
        self.c_proj = Conv1D(hidden_size, out_channels)
        # self.gelu = Gelu()

        self.layernorm = LayerNorm(in_channels=in_channels)
        self.residual_connect = ResidualConnection(dropout_prob=hidden_dropout)
        self.gelu_act = P.GeLU()
        self.dropout = nn.Dropout(1 - hidden_dropout)
        self.use_dropout = hidden_dropout > 0
        self.reshape = P.Reshape()

    def construct(self, input_tensor):
        """
        FeedForward construct function with layernorm and residual connection.

        Args:
            input_tensor (Tensor): the input of FeedForward layer, shape with [batch_szie * seq_len, n_embed].

        Returns:
            output (Tensor): the output of FeedForward layer, shape with [batch_szie * seq_len, n_embed]
        """
        # Feed Forward
        output = self.c_fc(input_tensor)  # [batch_szie * seq_len, n_embed * 4]
        output = self.gelu_act(output)
        # output = self.gelu(output)
        output = self.c_proj(output)  # [batch_szie * seq_len, n_embed]
        if self.use_dropout:
            output = self.dropout(output)
        # LayerNorm
        output = self.layernorm(output)

        # Add
        output = self.residual_connect(output, input_tensor)
        return output


class MaskedMultiHeadAttention(nn.Cell):
    """
    Masked multi-head attention block.
    """

    def __init__(self,
                 batch_size=512,
                 seq_length=2014,
                 n_embed=768,
                 n_head=12,
                 attention_dropout=0.02,
                 hidden_dropout=0.1,
                 has_attention_mask=True,
                 compute_type=mstype.float16
                 ):
        super(MaskedMultiHeadAttention, self).__init__()
        if n_embed % n_head != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (n_embed, n_head))

        self.dim_per_head = int(n_embed / n_head)  # 64

        self.masked_self_attention = MaskedSelfAttention(
            batch_size=batch_size,
            n_embed=n_embed,
            seq_length=seq_length,
            n_head=n_head,
            dim_per_head=self.dim_per_head,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            attention_dropout=attention_dropout,
            compute_type=compute_type,
        )

        self.layer_norm = LayerNorm(in_channels=n_embed)
        self.residual_connection = ResidualConnection(hidden_dropout)

        self.reshape = P.Reshape()
        self.new_shape = (-1, n_embed)

    def construct(self, input_tensor, attention_mask):
        """
        do masked multi head self-attention with layernorm and residual_connection.

        Args:
            input_tensor (Tensor): the embedding matrix of input sequence tokens,
                                   shape with [batch_size * seq_length, d_mdoel]
            attention_mask (Tensor): mask to avoid performing attention on padding token indices,
                                     shape with [batch_size, seq_len, seq_len].

        Returns:
            outputs (Tensor): the output of MaskedMultiHeadAttention, shape with [batch_size * seq_len, n_embed].
        """
        # LayerNorm
        output_tensor = input_tensor
        # masked multi-head attention
        # attention_output shape [batch_size * seq_length, n_embed]
        attention_output = self.masked_self_attention(output_tensor, attention_mask)
        output = self.residual_connection(attention_output, input_tensor)
        output = self.layer_norm(output)
        return output


class DecoderBlock(nn.Cell):
    """
    decoder block used in GPT2.

    Args:
        batch_size (int): Batch size of input dataset. Default: 512.
        seq_length (int): Length of input sequence. Default: 1024.
        n_embed (int): Size of the GPT2 decoder layers. Default: 768.
        n_head (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 3072.
        attention_dropout (float): The dropout probability for MaskedMultiHeadAttention. Default: 0.02.
        hidden_dropout (float): The dropout probability for hidden outputs. Default: 0.1.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: True.
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size=512,
                 seq_length=1024,
                 n_embed=768,
                 n_head=12,
                 intermediate_size=3072,
                 attention_dropout=0.02,
                 hidden_dropout=0.1,
                 has_attention_mask=True,
                 compute_type=mstype.float16
                 ):
        super(DecoderBlock, self).__init__()
        if n_embed % n_head != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (n_embed, n_head))

        self.dim_per_head = int(n_embed / n_head)  # 64

        self.masked_multi_head_attention = MaskedMultiHeadAttention(
            batch_size=batch_size,
            seq_length=seq_length,
            n_embed=n_embed,
            n_head=n_head,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            has_attention_mask=has_attention_mask,
            compute_type=compute_type
        )
        self.feedforward = FeedForward(
            in_channels=n_embed,
            out_channels=n_embed,
            hidden_size=intermediate_size,
            hidden_dropout=hidden_dropout
        )

        self.reshape = P.Reshape()
        self.new_shape = (-1, n_embed)

    def construct(self, input_tensor, attention_mask):  # input tensor shape[batch_size, seq_length, n_embed]
        """
        DecoderBlock with masked_multi_head_attention and feedforward.
        Args:
            input_tensor (Tensor): the embedding matrix of input sequence tokens,
                                   shape with [batch_size * seq_length, d_mdoel]
            attention_mask (Tensor): mask to avoid performing attention on padding token indices,
                                     shape with [batch_size, seq_len, seq_len].

        Returns:
            outputs (Tensor): the output of DecoderBlock, shape with [batch_size * seq_len, n_embed].
        """
        input_tensor = self.reshape(input_tensor, self.new_shape)

        # masked multi head attention with ln, res
        attention_output = self.masked_multi_head_attention(input_tensor, attention_mask)
        # feed forward with ln, res
        output = self.feedforward(attention_output)

        return output


class GPT2Transformer(nn.Cell):
    """
    Multi-layer GPT2 transformer.

    Args:
        batch_size (int): Batch size of input dataset. Default: 512.
        n_embed (int): Size of the decoder layers. Default: 768.
        seq_length (int): Length of input sequence. Default: 1024.
        n_layer (int): Number of hidden layers in decoder cells. Default: 12.
        n_head (int): Number of attention heads in decoder cells. Default: 12.
        intermediate_size (int): Size of intermediate layer in decoder cells. Default: 3072.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: True.
        attention_dropout (float): The dropout probability for MaskedMultiHeadAttention. Default: 0.1.
        hidden_dropout (float): The dropout probability for GPT2Output. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size=512,
                 n_embed=768,
                 seq_length=1024,
                 n_layer=12,
                 n_head=12,
                 intermediate_size=3072,
                 has_attention_mask=True,
                 attention_dropout=0.1,
                 hidden_dropout=0.1,
                 compute_type=mstype.float16):
        super(GPT2Transformer, self).__init__()

        layers = []
        for _ in range(n_layer):
            layer = DecoderBlock(batch_size=batch_size,
                                 seq_length=seq_length,
                                 n_embed=n_embed,
                                 n_head=n_head,
                                 intermediate_size=intermediate_size,
                                 attention_dropout=attention_dropout,
                                 hidden_dropout=hidden_dropout,
                                 has_attention_mask=has_attention_mask,
                                 compute_type=compute_type)
            layers.append(layer)

        self.layers = nn.CellList(layers)

        self.reshape = P.Reshape()
        self.new_shape = (-1, n_embed)
        # self.out_shape = (batch_size, seq_length, n_embed)
        self.out_shape = (-1, seq_length, n_embed)

    def construct(self, input_tensor, attention_mask):
        """
        Do Multi DecoderBlock.

        Args:
            input_tensor (Tensor): the embedding matrix of input sequence tokens,
                                   shape with [batch_size * seq_length, d_mdoel]
            attention_mask (Tensor): mask to avoid performing attention on padding token indices,
                                     shape with [batch_size, seq_len, seq_len].

        Returns:
            outputs (Tensor): the output of GPT2Transformer, shape with [batch_size * seq_len, n_embed].
        """
        prev_output = self.reshape(input_tensor, self.new_shape)
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask)
            prev_output = layer_output

        output = self.reshape(prev_output, self.out_shape)
        return output


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for GPT2Model.
    """

    def __init__(self, config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.input_mask = None
        self.compute_type = config.compute_type
        self.batch = config.batch_size

        assert self.input_mask_from_dataset

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.matmul = P.BatchMatMul()
        self.multiply = P.Mul()

        # mask future positions
        ones = np.ones(shape=(config.batch_size, config.seq_length, config.seq_length))
        self.lower_triangle_mask = Tensor(np.tril(ones), dtype=mstype.float32)

    def construct(self, input_mask, mask_future=True):
        """
        Construct network.

        Args:
            input_mask (Tensor): Tensor mask vectors with shape [batch_size, seq_len].
            mask_future (bool): Whether mask future (for decoder training). Default: True.

        Returns:
            attention_mask (Tensor): shape [batch_size, seq_len, seq_len].
        """
        attention_mask = input_mask.view(self.batch, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = self.cast(attention_mask, mstype.float32)

        return attention_mask


class GPT2Model(nn.Cell):
    """
    Decoder Representations from Transformers.

    Args:
        config (Class): Configuration for GPT2Model.
        is_training (bool): True for training mode. False for eval mode.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """

    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=False
                 ):
        super(GPT2Model, self).__init__()
        self.config = copy.deepcopy(config)
        if not is_training:
            self.config.hidden_dropout = 0.0
            self.config.attention_dropout = 0.0

        self.input_mask_from_dataset = self.config.input_mask_from_dataset
        self.batch_size = self.config.batch_size
        self.seq_length = self.config.seq_length
        self.n_embed = self.config.n_embed
        self.n_layer = self.config.n_layer
        self.embedding_dim = self.config.n_embed
        self.n_head = self.config.n_head
        self.size_per_head = self.config.size_per_head
        self.intermediate_size = self.config.intermediate_size
        if self.size_per_head is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError("The hidden size (%d) is not a multiple of the number "
                                 "of attention heads (%d)" % (self.hidden_size, self.num_attention_heads))
            else:
                self.size_per_head = int(self.hidden_size / self.num_attention_heads)
        else:
            assert isinstance(self.size_per_head, int)
            self.size_per_head = self.size_per_head

        self.last_idx = self.n_layer - 1

        self.gpt2_embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embedding_dim=self.embedding_dim,
            use_one_hot_embeddings=use_one_hot_embeddings,
            compute_type=config.compute_type
        )
        self.gpt2_embedding_postprocess = EmbeddingPostprocessor(
            embedding_dim=self.embedding_dim,
            seq_length=self.seq_length,
            n_positions=config.n_positions,
            dropout_prob=config.hidden_dropout
        )
        self.gpt2_decoder = GPT2Transformer(
            batch_size=self.batch_size,
            n_embed=self.n_embed,
            seq_length=self.seq_length,
            n_layer=self.n_layer,
            n_head=config.n_head,
            intermediate_size=config.intermediate_size,
            has_attention_mask=True,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            compute_type=config.compute_type
        )

        self.cast_compute_type = CastWrapper(dst_type=config.compute_type)
        self.layer_norm = LayerNorm(in_channels=self.n_embed)
        self.dropout = nn.Dropout(1 - config.hidden_dropout)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

        self.reshape = P.Reshape()
        self.new_shape = (-1, self.n_embed)

    def construct(self, input_ids, input_mask):
        """
        Construct network.

        Args:
            input_ids (Tensor): input sentences with shape [batch_size, seq_len].
            input_mask (Tensor): input sentences padding mask with shape [batch_size, seq_len],
                where 0 indicates padding position.

        Returns:
            decoder_output (Tensor): shape[batch_size, seq_len, n_embed].
            embedding_tables (Tensor): word embeddings with shape [vocab_size, n_embed]
        """
        # Embedding
        word_embeddings, embedding_tables = self.gpt2_embedding_lookup(input_ids)
        embedding_output = self.gpt2_embedding_postprocess(word_embeddings)
        embedding_output = self.dropout(embedding_output)

        # Attention mask with shape [batch_size, seq_len, seq_len]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask, True)
        # GPT2 decoder
        decoder_output = self.gpt2_decoder(
            self.cast_compute_type(embedding_output),
            self.cast_compute_type(attention_mask)
        )

        # LayerNorm
        decoder_output = self.reshape(decoder_output, self.new_shape)
        decoder_output = self.layer_norm(decoder_output)
        decoder_output = self.reshape(decoder_output, (-1, self.seq_length, self.n_embed))

        return decoder_output, embedding_tables

    def get_token_embeddings(self):
        return self.gpt2_embedding_lookup.embedding_table.asnumpy()
