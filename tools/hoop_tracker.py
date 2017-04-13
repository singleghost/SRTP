#!/usr/bin/env python
#coding=utf-8
import numpy as np
import cv2
import cv2.cv as cv
import sys

help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt

    cascade_fn = "../classifier/cascade.xml"

    video_path = sys.argv[1]
    cascade = cv2.CascadeClassifier(cascade_fn)
    print video_path
    cam = cv2.VideoCapture(video_path)
    if cam.isOpened():
	    print "视频正常打开"
    else:
	    print "视频没有正常打开"
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = cv2.getTickCount()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        dt = (cv2.getTickCount() - t) / cv2.getTickFrequency()

        print 'time: %.1f ms' % (dt*1000)
        cv2.imshow('hoopdetect', vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
