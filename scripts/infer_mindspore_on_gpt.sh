#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash infer_mindspore.sh SENTENCE MS_GPT_PATH KIND DEVICE_TARGET"
echo "for example: bash infer_mindspore_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_base2medium_aki.ckpt" --kind="gpt_medium" --DEVICE_TARGET="CPU""
echo "=============================================================================================================="

SENTENCE=$1
MS_GPT_PATH=$2
KIND=$3
DEVICE_TARGET=$4
python infer_on_gpt.py\
   --sentence="${SENTENCE}"\
   --ms_gpt_path=$MS_GPT_PATH\
   --kind=$KIND\
   --device_target=$DEVICE_TARGET
echo press any key to continue
read -n 1
echo Continue running