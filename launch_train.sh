#!/bin/bash
MODEL=$1
GPU=$2
shift 2

mkdir -p logs
LOG_FILE="${MODEL//\//\_}.log"