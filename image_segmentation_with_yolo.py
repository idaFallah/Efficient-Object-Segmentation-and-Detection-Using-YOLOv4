# downloading darknet

!git clone https://github.com/AlexeyAB/darknet

ls  # showing all the folders in the currenet directory

cd darknet/

ls

# compiling the lib

!make

# downloading YOLO weights (for transfer learning)

!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# testing the object detector

ls

!./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg

import cv2
import matplotlib.pyplot as plt
def show_detection(path):
  image = cv2.imread(path)
  fig = plt.gcf()  # gcf will get the axis & start in the figure objects in order to show the image
  fig.set_size_inches(18, 10)
  plt.axis('off')
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

show_detection('predictions.jpg')

# darknet with gpu

import tensorflow as tf
tf.test.gpu_device_name()

ls

# makefile has the info abut how we compile the library

!sed -i 's/OPENCV=0/OPENCV=1/' Makefile # wether we use opencv
!sed -i 's/GPU=0/GPU=1/' Makefile  # wether we use gpu
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile  # type of gpu

!make # to compile the library

!./darknet detect cfg/yolov4.cfg yolov4.weights data/giraffe.jpg

show_detection('predictions.jpg')

# coco names contains all the classes that we can classify with this library
# we can also use imagenet objects, besided coco dataset objects



# quality of detections(threshold)

!./darknet detect cfg/yolov4.cfg yolov4.weights data/horses.jpg

show_detection('predictions.jpg')

!./darknet detect cfg/yolov4.cfg yolov4.weights data/horses.jpg -thresh 0.9

show_detection('predictions.jpg')

# we use threshold when we want to be sure about detection of an object

!./darknet detect cfg/yolov4.cfg yolov4.weights data/horses.jpg -ext_output
# shows more detail = the values of the binding boxes

# objetc detection in videos

from google.colab import drive
drive.mount('/content/drive ')

ls

!./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show /content/drive/MyDrive/Computer\ Vision\ /Videos/video_street2.mp4 -i 0 -out_filename /content/video_street2_FResult.avi

!ls /content/drive/MyDrive/Computer\ Vision\ /Videos/









