#!/bin/bash
mkdir -p data
mkdir -p data/input
mkdir -p data/output
mkdir -p logs
mkdir -p notebooks
mkdir -p pickle

touch configs/common/notify.yml

kaggle competitions download -c bengaliai-cv19 -p ./data/input/