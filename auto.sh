#!/bin/bash

find ./positive_images -iname "*.jpg" > positives.txt
find ./negative_images -iname "*.jpg" > negatives.txt
find ./positive_images -name '*.jpg' -exec echo \{\} 1 0 0 40 40 \; > samplesdescription.dat

opencv_createsamples -info samplesdescription.dat -vec samples.vec -w 40 -h 40

opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
  -numStages 20 -minHitRate 0.998 -maxFalseAlarmRate 0.5 -numPos 600\
  -numNeg 5500 -w 40 -h 40 -mode ALL -precalcValBufSize 1024\
  -precalcIdxBufSize 1024
