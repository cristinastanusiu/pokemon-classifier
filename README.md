*1)* The pokemon images have been obtained and annotated manually using imgLabelGUI.

*2)* Dependencies - Cuda+Graphics card, OpenCV(Optional)

run Makefile to compile the code.
Use GPU=1 for using the GPU before Make.
Set the correct path to cuda libraries to overcome compilation errors.

Important files inside root directory

src contains the source code for running darknet-yolo
Images folder contains all the images for training.
Labels folder contains the annotated data of each image.
train.txt file contains paths to images used for training. Only images for which path is provided will be trained.
cfg/yolo.cfg file is used as configuration for Neural Net.
data/labels contains png files which are used in tagging bounding boxes.
backup/yolo_final.weights is the final trained weights.

See yolo.c for configurations regarding using darknet with different directory and class configurations.

#*Training*

`./darknet yolo train <config-file> <pre-dev-weights>`

`./darknet yolo train cfg/yolo.cfg extraction.conv.weights`

#*Testing on image*

`./darknet yolo test <config-file> <trained-weights> <path-to-image-testing> -thresh 0.2 [Optional-default value 0.25]`

`./darknet yolo test cfg/yolo.cfg backup/yolo_final.weights images/charizard_305.jpg -thresh 0.2`

#*Testing on video*

`./darknet yolo demo_vid cfg/yolo.cfg backup/yolo_final.weights 3.mp4 -thresh 0.1`

see predictions.png or out.avi for the result of testing.


#*Multi-class SVM for pokemon detection*

`python main.py`



github repository : https://github.com/mayukuse24/pokemon-classifier

Dataset at : https://drive.google.com/open?id=0B7ToU0uexZWkeE5RVC1rWkMteDQ
