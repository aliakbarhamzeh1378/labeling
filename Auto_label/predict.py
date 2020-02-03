#! /usr/bin/env python

import json
import os
import xml.etree.cElementTree as ET

import cv2

from Auto_label.frontend import YOLO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def label(path, weights_path, config_path):
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    yolo = YOLO(backend=config['model']['backend'],
                input_sizeW=config['model']['input_sizeW'],
                input_sizeH=config['model']['input_sizeH'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])
    yolo.load_weights(weights_path)
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            print(file)
            image = cv2.imread(path + file)
            boxes = yolo.predict(image)

            image_h, image_w, chanel = image.shape

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = path.split("/")[-2]
            ET.SubElement(root, "filename").text = file
            ET.SubElement(root, "path").text = path + file

            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = "Unknown"

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(image_w)
            ET.SubElement(size, "height").text = str(image_h)
            ET.SubElement(size, "depth").text = str(chanel)

            ET.SubElement(root, "segmented").text = "0"
            ObjectsFound = []
            for box in boxes:

                xmin = int(box.xmin * image_w)
                ymin = int(box.ymin * image_h)
                xmax = int(box.xmax * image_w)
                ymax = int(box.ymax * image_h)

                labels = config['model']['labels']
                sObj = labels[box.get_label()] + ' ' + str(box.get_score())
                if labels[box.get_label()] == "None_Object": continue
                if xmin <= 0: continue
                # cv2.rectangle(image_color, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                if (box.get_score() > 0.6):
                    ObjectsFound.append([labels[box.get_label()], xmin, ymin, box.get_score()])
                    # cv2.putText(image_color, sObj, (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

                object = ET.SubElement(root, "object")
                ET.SubElement(object, "name").text = labels[box.get_label()]
                ET.SubElement(object, "pose").text = "Unspecified"
                ET.SubElement(object, "truncated").text = "0"
                ET.SubElement(object, "difficult").text = "0"
                bndbox = ET.SubElement(object, "bndbox")

                ET.SubElement(bndbox, "xmin").text = str(xmin - 5)
                ET.SubElement(bndbox, "ymin").text = str(ymin - 5)
                ET.SubElement(bndbox, "xmax").text = str(xmax + 5)
                ET.SubElement(bndbox, "ymax").text = str(ymax + 5)
            tree = ET.ElementTree(root)
            tree.write(path + file[:-3] + 'xml')

