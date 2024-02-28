import mindspore.common.dtype as mstype


class GPT2Config:
    """
       Configuration for `GPT2Model`.

       Args:
           batch_size (int): Batch size of input dataset. Default: 512.
           seq_length (int): Length of input sequence. Default: 1024.
           vocab_size (int): The shape of each embedding vector. Default: 50257.
           n_embed (int): Size of the bert encoder layers. Default: 768.
           n_layer (int): Number of hidden layers in the GPT2Transformer decoder block. Default: 12.
           n_head (int): Number of attention heads in the GPT2Transformer decoder block. Default: 12.
           intermediate_size (int): Size of intermediate layer in the GPT2Transformer decoder block. Default: 3072.
           hidden_act (str): Activation function used in the GPT2Transformer decoder block. Default: "gelu".
           hidden_dropout (float): The dropout probability for GPT2Output. Default: 0.1.
           attention_dropout (float): The dropout probability for MaskedMultiHeadAttention. Default: 0.1.
           n_positions (int): Maximum length of sequences used in this model. Default: 1024.
           initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
           input_mask_from_dataset (bool): Specifies whether to use the input mask that loaded from dataset.
                                           Default: True.
           summary_first_dropout (float): The dropout probability for GPT2CBTModel. Default: 0.1.
           dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
           compute_type (:class:`mindspore.dtype`): Compute type in GPT2Transformer. Default: mstype.float16.
       """

    def __init__(self,
                 batch_size=512,
                 seq_length=1024,
                 vocab_size=50257,
                 n_embed=768,
                 n_layer=12,
                 n_head=12,
                 size_per_head=None,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout=0.1,
                 attention_dropout=0.1,
                 n_positions=1024,
                 initializer_range=0.02,
                 input_mask_from_dataset=True,
                 summary_first_dropout=0.1,
                 dtype=mstype.float32,
                 compute_type=mstype.float16,
                 ):
        self.size_per_head = size_per_head
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.n_positions = n_positions
        self.initializer_range = initializer_range
        self.input_mask_from_dataset = input_mask_from_dataset
        self.summary_first_dropout = summary_first_dropout
        self.dtype = dtype
        self.compute_type = compute_type
        if self.size_per_head is None:
            self.size_per_head = self.n_embed // self.n_head
