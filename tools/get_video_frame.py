#!/usr/bin/env python
#coding=utf-8
import numpy as np
import sys
import cv2

vc = cv2.VideoCapture(sys.argv[1])#读入视频文件

if vc.isOpened(): #判断是否正常打开
	totalFrameNumber = vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	print "totalFrameNumber", totalFrameNumber
	print "fps:", vc.get(cv2.cv.CV_CAP_PROP_FPS)
else:
	print "视频没有正常打开"
	sys.exit(1)

fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
		         int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

frametostart=0

vc.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frametostart)

frametostop=frametostart+50000000
c=frametostart
unstop=1

rval, frame = vc.read()
cols = frame.shape[1]	
rows = frame.shape[0]
while unstop:
	if c % 50000 == 0:
		rval ,frame=vc.read()
		if rval == 0:
			print "read video failure"
			break
		#frame = frame[ rows*2/5:rows, 0:cols]
		cv2.imwrite('../test_images/test_img'+str(c) + '.jpg',frame)
	c=c+1
	if c>frametostop:
		unstop=0
	

vc.release()
cv2.destroyAllWindows()
