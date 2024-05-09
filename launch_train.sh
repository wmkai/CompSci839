#!/bin/bash
MODEL=$1
GPU=$2
shift 2

mkdir -p logs
LOG_FILE="${MODEL//\//\_}.log"
LOG_FILE="logs/${LOG_FILE}"
echo "Training ${MODEL} on GPU ${GPU}"
echo "${LOG_FILE}"
nohup bash train.sh </dev/null > "${LOG_FILE}" 2>&1 &
