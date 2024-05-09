#!/bin/bash
MODEL=$1
GPU=$2
shift 2

mkdir -p logs
LOG_FILE="${MODEL//\//\_}.log"
LOG_FILE="logs/${LOG_FILE}"
echo "Logging to ${LOG_FILE}"
nohup bash python simple_train.py --model $MODEL --device $GPU $@ </dev/null > "${LOG_FILE}" 2>&1 &
