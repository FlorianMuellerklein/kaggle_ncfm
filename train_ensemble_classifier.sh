#!/bin/bash

for i in {1..30}
do
    python train_classifier_ensemble.py $i
done
