import os.path
import pickle

from transformers import BertTokenizer


def get_sst_tokenizer():
    if not os.path.exists("./SST-2/vocab.txt"):
        build_vocab()
    return BertTokenizer("./SST-2/vocab.txt")


def build_vocab():
    f = open('./SST-2/vocab.pkl', 'rb')
    data = pickle.load(f)
    with open("SST-2/vocab.txt", "w", encoding='utf-8') as f:
        f.write("[UNK]\n")
        f.write("[CLS]\n")
        f.write("[SEP]\n")
        f.write("[MASK]\n")
        for key in data:
            f.write(key + "\n")
