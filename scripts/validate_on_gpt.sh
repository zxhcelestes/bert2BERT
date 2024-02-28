#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash validate.sh TASK_TYPE METHOD"
echo "for example: bash validate_on_gpt.sh base2medium fpi"
echo "=============================================================================================================="

TASK_TYPE=$1
METHOD=$2
python eval_on_gpt.py\
   --task_type=$TASK_TYPE\
   --method=$METHOD
echo press any key to continue
read -n 1
echo Continue running