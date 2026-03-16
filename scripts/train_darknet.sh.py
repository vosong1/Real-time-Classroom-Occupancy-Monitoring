#!/bin/bash

DATA=configs/classroom.data
CFG=configs/yolov3-classroom.cfg
PRETRAIN=darknet53.conv.74

./darknet detector train $DATA $CFG $PRETRAIN -map