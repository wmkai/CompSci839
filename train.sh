#!/bin/bash
MODEL=$1
GPU=$2
shift 2

CUDA_VISIBLE_DEVICES=$GPU python train.py $MODEL $@