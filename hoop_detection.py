#coding=utf-8
import cv2
import numpy as np

hoop_cascade = cv2.CascadeClassifier('classifier/cascade.xml')

test_image_set = ['NBA_image1.png', 'NBA_image2.png', 'NBA_image3.png', 
		'NBA_image4.png', 'NBA_image5.png', 'NBA_image6.png']

test_image_set = [ 'test_images/' + x for x in test_image_set ]
i = 0
for image_name in test_image_set:
	img = cv2.imread(image_name)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = hoop_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
	
	cv2.imshow('img' + str(i), img)
	i = i + 1
cv2.waitKey(0)
cv2.destroyAllWindows()



