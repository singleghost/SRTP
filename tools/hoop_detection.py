#coding=utf-8
import cv2
import numpy as np
import sys
import os
from time import sleep

hoop_cascade = cv2.CascadeClassifier('classifier/cascade.xml')

#test_image_set = ['NBA_image1.png', 'NBA_image2.png', 'NBA_image3.png', 
#		'NBA_image4.png', 'NBA_image5.png', 'NBA_image6.png']

_dir = sys.argv[1]
test_image_set = os.listdir(_dir)
test_image_set = [ x for x in test_image_set if x.endswith(".jpg") ]

test_image_set = [ _dir + x for x in test_image_set ]
test_image_set = test_image_set[160:180]

test_labels = [0] * 265
test_labels += [1, 1, 1, 1, 1, ] + [1] * 43

i = 0

print test_image_set
for image_name in test_image_set:
	
	img = cv2.imread(image_name)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(gray, gray)
	hoops = hoop_cascade.detectMultiScale(gray, 1.3, 5)
	if len(hoops) > 0:
		for (x,y,w,h) in hoops:
			cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
	
	cv2.imshow('NBA_hoop' + str(i), img)
	i = i + 1

print "全部测试完成"
cv2.waitKey(0)
cv2.destroyAllWindows()


