#!/bin/bash
# Uncomment the <RECORD> to select the pretrained model
RECORD=0043
#RECORD=115

python3 pre_main_long.py --mode val --best 1 --record $RECORD

#tail record/$RECORD/log.txt -n 1
