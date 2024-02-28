import json
import os.path
import pickle

from transformers import GPT2Tokenizer


def get_sst_tokenizer():
    if not (os.path.exists("./SST-2/vocab.json") and os.path.exists("./SST-2/merges.txt")):
        build_vocab()
    return GPT2Tokenizer("./SST-2/vocab.json", "./SST-2/merges.txt")


def build_vocab():
    f = open('./SST-2/vocab.pkl', 'rb')
    data = pickle.load(f)
    with open("SST-2/merges.txt", "w", encoding='utf-8') as f:
        f.write("<|endoftext|>\n")
        f.write("<|endoftext|>\n")
        f.write("<|endoftext|>\n")
        # f.write("[MASK]\n")
        for key in data:
            f.write(key + "\n")
    # open txt file
    with open("SST-2/merges.txt", 'r', encoding='utf-8') as f:
        cnt = 0
        d = {}
        for line in f:
            # line represents a line in the txt file,
            # removing the newline after each line and using it as the value of the content key of dictionary d
            d[line.rstrip('\n')] = cnt
            cnt += 1
        # Create a json file with mode set to 'a'
        with open('SST-2/vocab.json', 'w', encoding='utf-8') as fe:
            json.dump(d, fe, ensure_ascii=False)
            # Write dictionary d to json file and set ensure_ascii = False,
            # mainly because Chinese characters are ascii characters,
            # If this parameter is not specified, the text format is ascii
