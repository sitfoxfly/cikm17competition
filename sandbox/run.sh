#!/bin/bash

python lazada.py ../data/training/data_train.csv ../data/training/clarity_train.labels ../data/validation/data_valid.csv clarity_valid.predict
python lazada.py ../data/training/data_train.csv ../data/training/conciseness_train.labels ../data/validation/data_valid.csv conciseness_valid.predict

zip valid.predict.zip *.predict