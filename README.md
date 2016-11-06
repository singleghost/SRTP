# Train your own OpenCV Haar classifier

**Important**: This guide assumes you work with OpenCV 2.4.x. Since I no longer work with OpenCV, and don't have the time to keep up with changes and fixes, this guide is **unmaintained**. Pull requests will be merged of course, and if someone else wants commit access, feel free to ask!

This repository aims to provide tools and information on training your own
OpenCV Haar classifier.  Use it in conjunction with this blog post: [Train your own OpenCV Haar
classifier](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html).



## Instructions

1. Install OpenCV & get OpenCV source

        brew tap homebrew/science
        brew install --with-tbb opencv
        wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip
        unzip opencv-2.4.9.zip

2. Clone this repository

        git clone https://github.com/mrnugget/opencv-haar-classifier-training

3. Put your positive images in the `./positive_images` folder and create a list
of them:

        find ./positive_images -iname "*.jpg" > positives.txt

4. Put the negative images in the `./negative_images` folder and create a list of them:

        find ./negative_images -iname "*.jpg" > negatives.txt

5. Create positive samples with the `bin/createsamples.pl` script and save them
to the `./samples` folder:

        perl bin/createsamples.pl positives.txt negatives.txt samples 1500\
          "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
          -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 80 -h 40"

6. Use `tools/mergevec.py` to merge the samples in `./samples` into one file:

        python ./tools/mergevec.py -v samples/ -o samples.vec
        
   Note: If you get the error `struct.error: unpack requires a string argument of length 12`
   then go into your **samples** directory and delete all files of length 0.

7. Start training the classifier with `opencv_traincascade`, which comes with
OpenCV, and save the results to `./classifier`:

        opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
          -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000\
          -numNeg 600 -w 80 -h 40 -mode ALL -precalcValBufSize 1024\
          -precalcIdxBufSize 1024

8. Wait until the process is finished (which takes a long time — a couple of
days probably, depending on the computer you have and how big your images are).

9. Use your finished classifier!

        cd ~/opencv-2.4.9/samples/c
        chmod +x build_all.sh
        ./build_all.sh
        ./facedetect --cascade="~/finished_classifier.xml"


## Acknowledgements

A huge thanks goes to Naotoshi Seo, who wrote the `mergevec.cpp` and
`createsamples.cpp` tools and released them under the MIT licencse. His notes
on OpenCV Haar training were a huge help. Thank you, Naotoshi!

## References & Links:

- [Naotoshi Seo - Tutorial: OpenCV haartraining (Rapid Object Detection With A Cascade of Boosted Classifiers Based on Haar-like Features)](http://note.sonots.com/SciSoftware/haartraining.html)
- [Material for Naotoshi Seo's tutorial](https://code.google.com/p/tutorial-haartraining/)
- [OpenCV Documentation - Cascade Classifier Training](http://docs.opencv.org/doc/user_guide/ug_traincascade.html)
# SRTP
