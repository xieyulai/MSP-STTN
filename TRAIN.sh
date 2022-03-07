#!/bin/bash
#  <RECORD> is the training and record ID
RECORD=3090

# Training
python3 pre_main_short.py --mode train --record $RECORD

# Training with a pretrained model
python3 pre_main_short.py --mode train --record $RECORD  --presume_record 3089 --presume_epoch_s 79 --keep_train 1

# Testing
python3 pre_main_short.py --mode val --record $RECORD 

#tail record/$RECORD/log.txt -n 1
