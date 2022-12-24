#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash infer_mindspore.sh SENTENCE MS_BERT_PATH KIND"
echo "for example: bash infer_mindspore_on_gpt.py --sentence="I like it" --ms_bert_path="D:\document\bert2BERT\output\gpt_base2medium_aki.ckpt" --kind="gpt_medium""
echo "=============================================================================================================="

SENTENCE=$1
MS_BERT_PATH=$2
KIND=$3
python infer_on_gpt.py\
   --sentence=SENTENCE\
   --ms_bert_path=$MS_BERT_PATH\
   --kind=$KIND
echo press any key to continue
read -n 1
echo Continue running