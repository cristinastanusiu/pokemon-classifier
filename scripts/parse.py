import sys
import xml.etree.ElementTree as ET

tree = ET.parse(sys.argv[1])
root = tree.getroot()

for i in root.findall("object"):
    j = i.find("bndbox")
    print i.find("name").text, j.find("xmin").text, j.find("ymin").text, int(j.find("xmax").text)-int(j.find("xmin").text), int(j.find("ymax").text) - int(j.find("ymin").text)
    
