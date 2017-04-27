#See VOCdevkit

import sys
import xml.etree.ElementTree as ET

classes = ["car","train","bus"]

data_folder = "/users/pritamvarma/VOCdevkit/VOC2007/"

image_loc = data_folder + "JPEGImages/"
cimage_folder = data_folder + "ImageSets/Main/"

image_list = []
#Get List of image names for classes

for class_name in classes :
    with open(cimage_folder + class_name+"_test.txt") as f:
        lines = f.read().splitlines()
        pres_list = [x.split(" ")[0] for x in lines if x[-2] != '-']
        print(pres_list)
        image_list += pres_list

print(image_list)

f = open('/users/pritamvarma/Darknet-Yolo-orig/train.txt', 'w')
for image in image_list :
    f.write(image_loc + image +'\n')  # python will convert \n to os.linesep
f.close()


#Convert Annotation .xml to .txt with w,h,x,y

annotated_loc = data_folder + "Annotations/"
for image in image_list :
    tree = ET.parse(annotated_loc+image+".xml")
    root = tree.getroot()

    for i in root.findall("object"):
        j = i.find("bndbox")
        print i.find("name").text, j.find("xmin").text, j.find("ymin").text, int(j.find("xmax").text)-int(j.find("xmin").text), int(j.find("ymax").text) - int(j.find("ymin").text)
