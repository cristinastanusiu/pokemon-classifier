![Darknet Logo](http://pjreddie.com/static/img/darknet.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

#About This Fork#

![Yolo logo](https://olimex.files.wordpress.com/2014/04/tv.png)

1. This fork repository adds some additional niche in addition to the current darkenet from pjreddie. e.g.

   (1). Read a video file, process it, and output a video with boundingboxes.
   
   (2). Some util functions like image_to_Ipl, converting the image from darknet back to Ipl image format from OpenCV(C).
   
   (3). Adds some python scripts to label our own data, and preprocess annotations to the required format by darknet.  
   
   ...More to be added

2. This fork repository illustrates how to train a customized neural network with our own data, with our own classes.

   The procedure is documented in README.md.
   

#DEMOS of YOLO trained with our own data#

TV : [https://www.youtube.com/watch?v=t4_O5Cs7uRM](https://www.youtube.com/watch?v=t4_O5Cs7uRM)

#How to Train With Customized Data and Class Numbers/Labels#

1. Collect Data and Annotation
   Crowd Sourcing by students
2. Create Annotation in Darknet Format 
   
   (1). If we choose to use VOC data to train, use [scripts/voc_label.py] to convert existing VOC annotations to darknet format.
   
   (2). If we choose to use our own collected data, use [scripts/convert.py](https://github.com/frankzhangrui/Darknet-Yolo/blob/master/convert.py) to convert the annotations.

   At this step, we should have darknet annotations(.txt) and a training list(.txt).
   
   Upon labeling, the format of annotations generated by [BBox-Label-Tool](https://github.com/puzzledqs/BBox-Label-Tool) is:
   
   class_number
   
   box1_x1 box1_y1 box1_width box1_height
   
   box2_x1 box2_y1 box2_width box2_height
   
   ....
   
   After conversion, the format of annotations converted by [scripts/convert.py](https://github.com/Guanghan/darknet/blob/master/scripts/convert.py) is:
   
   class_number box1_x1_ratio box1_y1_ratio box1_width_ratio box1_height_ratio
   
   class_number box2_x1_ratio box2_y1_ratio box2_width_ratio box2_height_ratio
   
   ....
   
   Note that each image corresponds to an annotation file. But we only need one single training list of images. Remember to put the folder "images" and folder "labels" in the same parent directory, as the darknet code look for annotation files this way (by default). 
   
   
3. Modify Some Code

   (1) In [src/yolo.c](https://github.com/Guanghan/darknet/blob/master/src/yolo.c), change class numbers and class names. (And also the paths to the training data and the annotations, i.e., the list we obtained from step 2. )
   
       If we want to train new classes, in order to display correct png Label files, we also need to moidify and run [data/labels/make_labels] (https://github.com/Guanghan/darknet/blob/master/data/labels/make_labels.py)
   
   (2) In [src/yolo_kernels.cu](https://github.com/Guanghan/darknet/blob/master/src/yolo_kernels.cu), change class numbers.
   
   (3) Now we are able to train with new classes, but there is one more thing to deal with. In YOLO, the number of parameters of the second last layer is not arbitrary, instead it is defined by some other parameters including the number of classes, the side(number of splits of the whole image). Please read [the paper](http://arxiv.org/abs/1506.02640)  
       
       (5 x 2 + number_of_classes) x 11 x 11, as an example, assuming no other parameters are modified.  
       
       Therefore, in [cfg/yolo.cfg](https://github.com/Guanghan/darknet/blob/master/cfg/yolo.cfg), change the "output" in line 218, and "classes" in line 222.
       
   (4) Now we are good to go. If we need to change the number of layers and experiment with various parameters, just mess with the cfg file. For the original yolo configuration, we have the [pre-trained weights](http://pjreddie.com/media/files/extraction.conv.weights) to start from. For arbitrary configuration, I'm afraid we have to generate pre-trained model ourselves.
   
4. Start Training

   Try something like:

   ./darknet yolo train cfg/yolo.cfg extraction.conv.weights

# Darknet-Yolo
Object Detection End-to-End framework