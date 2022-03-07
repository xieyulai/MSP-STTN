#!/bin/bash
# Uncomment the <RECORD> to select the pretrained model
#STEP1
RECORD=104
#RECORD=3089
#RECORD=5042
#STEP2
#RECORD=5047
#STEP3
#RECORD=5050
#STEP4
#RECORD=0051
#STEP5
#RECORD=2051
#STEP5
#RECORD=1051

DATASET=All

IS_RECT=1

python3 pre_main_short.py --mode val --best 1 --record $RECORD --is_rect $IS_RECT

#tail record/$RECORD/log.txt -n 1
