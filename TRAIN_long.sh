#!/bin/bash
#  <RECORD> is the training and record ID
RECORD=5061

# Training
python3 pre_main_long.py --mode train --record $RECORD

# Testing
python3 pre_main_long.py --mode val --record $RECORD

#tail record/$RECORD/log.txt -n 1
