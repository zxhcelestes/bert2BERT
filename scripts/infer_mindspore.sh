#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash infer_mindspore.sh SENTENCE MS_BERT_PATH KIND DEVICE_TARGET"
echo "for example: bash infer.py --sentence="I like it" --ms_bert_path="./output/bert_small2base_aki.ckpt" --kind="bert_base" --DEVICE_TARGET="CPU""
echo "=============================================================================================================="

SENTENCE=$1
MS_BERT_PATH=$2
KIND=$3
DEVICE_TARGET=$4
python infer.py --sentence="${SENTENCE}" --ms_bert_path=$MS_BERT_PATH --kind=$KIND --device_target=$DEVICE_TARGET
echo press any key to continue
read -n 1
echo Continue running