#!/bin/bash

for i in {0..4}
do
  echo '-----------------------------------------------------------------------------------'
  echo 'Training Net:'
  echo $i
  echo '-----------------------------------------------------------------------------------'
  python train_resnet_fullimg_classifier.py $i
  python train_locnet_ensmb.py $i 40
done
