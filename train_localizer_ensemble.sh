#!/bin/bash

for i in {0..4}
do
    python train_locnet_split.py $i
done
