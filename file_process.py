import os
import xml.etree.cElementTree as ET

import cv2
from tqdm import tqdm

from Auto_label.predict import label


def walkdir(folder):
    """Walk through each files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))


def add_prefix(path, prefix=None):
    if path[-1] != "/": path = path + "/"
    for file in os.listdir(path):
        os.rename(path + file, path + prefix + "_" + file)


def get_folders(path):
    if path[-1] != "/": path = path + "/"
    folders = []
    for file in os.listdir(path):
        if os.path.isdir(path + file):
            folders.append(file)
    return folders


def bndbox_change(tree, pl=10, pt=10, pr=10, pb=10, label=[]):
    flag = False
    for i in tree.iter('object'):
        if i.find('name').text in label:
            flag = True
            xmin = i.find('bndbox/xmin').text
            ymin = i.find('bndbox/ymin').text
            xmax = i.find('bndbox/xmax').text
            ymax = i.find('bndbox/ymax').text
            i.find('bndbox/xmin').text = str(int(xmin) + pl)
            i.find('bndbox/ymin').text = str(int(ymin) + pt)
            i.find('bndbox/xmax').text = str(int(xmax) + pr)
            i.find('bndbox/ymax').text = str(int(ymax) + pb)

    return tree, flag


def get_label(path):
    if path[-1] != "/": path = path + "/"
    label = []
    for file in os.listdir(path):
        if file.endswith(".xml"):
            tree = ET.parse(path + '/' + file)
            for i in tree.iter('object'):
                if not i.find('name').text in label:
                    label.append(i.find('name').text)
    return label


def get_count_label(path):
    if path[-1] != "/": path = path + "/"
    labels = {}
    for file in os.listdir(path):
        if file.endswith(".xml"):
            tree = ET.parse(path + '/' + file)
            for i in tree.iter('object'):
                if not i.find('name').text in labels.keys():
                    labels[i.find('name').text] = 1
                else:
                    count = labels[i.find('name').text]
                    labels[i.find('name').text] = count + 1

    return labels


def get_file_label(path, label):
    labels = []
    for file in os.listdir(path):
        if file.endswith(".xml"):
            tree = ET.parse(path + '/' + file)
            for i in tree.iter('object'):
                if i.find('name').text == label:
                    labels.append(file)

    return labels


def label_change(path, src, dst):
    count = 0
    if path[-1] != "/": path = path + "/"
    for file in os.listdir(path):
        tree = ET.parse(path + file)
        for i in tree.iter('object'):
            if i.find('name').text == src:
                count += 1
                i.find('name').text = dst

        tree.write(path + file)
    return count


def labeler(path, weights_path, config_path):
    label(path,
          weights_path=weights_path,
          config_path=config_path)


def remove_bad_xml(Color_path):
    if Color_path[-1] != "/": Color_path = Color_path + "/"

    for file in os.listdir(Color_path):
        if file.endswith(".xml"):
            tree = ET.parse(Color_path + file)
            obj = tree.find("object")
            if obj is not None:
                continue
            else:
                # Remove XML
                os.remove(Color_path + file)
                os.remove(Color_path.replace('Color', 'Dac') + file)
                os.remove(Color_path.replace('Color', 'DacColor') + file)
                # Remove Image
                os.remove(Color_path + file[:-3] + 'jpg')
                os.remove(Color_path.replace('Color', 'Dac') + file[:-3] + 'jpg')
                os.remove(Color_path.replace('Color', 'DacColor') + file[:-3] + 'jpg')


def remove_image_with_no_xml(path):
    if path[-1] != "/": path = path + "/"
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            if not os.path.exists(path + file[:-3] + 'xml'):
                os.remove(path + file)


def remove_xml_with_no_image(path):
    if path[-1] != "/": path = path + "/"
    for file in os.listdir(path):
        if file.endswith(".xml"):
            if not os.path.exists(path + file[:-3] + 'jpg'):
                os.remove(path + file)


def convert_to_voc(img_path, annon_path):
    image_type = 'jpg'
    classes = get_label(annon_path)
    list_file = open('data.txt', 'w')
    for image in os.listdir(img_path):
        list_file.write(img_path + image)
        xml = annon_path + image[:-len(image_type)] + 'xml'
        convert_annotation_to_voc(xml, list_file, classes)
        list_file.write('\n')
    list_file.close()
    pass


def convert_annotation_to_voc(xml_file_path, list_file, classes):
    file = open(xml_file_path)
    tree = ET.parse(file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def crop_from_xml(image_path, label_path, save_path):
    if image_path[-1] != "/": image_path = image_path + "/"
    if label_path[-1] != "/": label_path = label_path + "/"
    if save_path[-1] != "/": save_path = save_path + "/"

    for file in tqdm(walkdir(label_path), total=len(os.listdir(label_path)), unit="files"):
        file = file.split("/")[-1]
        if file.endswith(".xml"):

            if os.path.exists(image_path + file[:-3] + 'jpg'):
                tree = ET.parse(label_path + file)
                obj = tree.find('object')
                xmin = obj.find('bndbox/xmin').text
                ymin = obj.find('bndbox/ymin').text
                xmax = obj.find('bndbox/xmax').text
                ymax = obj.find('bndbox/ymax').text
                img = cv2.imread(image_path + file[:-3] + 'jpg')
                crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                cv2.imwrite(save_path + file[:-3] + 'jpg', crop_img)


def resize_image_label(image_path, label_path, width=640, height=480, xml_save_path="", image_save_path=""):
    if image_path[-1] != "/": image_path = image_path + "/"
    if label_path[-1] != "/": label_path = label_path + "/"
    if xml_save_path[-1] != "/": xml_save_path = xml_save_path + "/"
    if image_save_path[-1] != "/": image_save_path = image_save_path + "/"
    if not os.path.exists(xml_save_path): os.mkdir(xml_save_path)
    if not os.path.exists(image_save_path): os.mkdir(image_save_path)
    for file in tqdm(walkdir(label_path), total=len(os.listdir(label_path)), unit="files"):
        file = file.split("/")[-1]
        if file.endswith(".xml"):
            if os.path.exists(image_path + file[:-3] + 'jpg'):
                img = cv2.imread(image_path + file[:-3] + 'jpg')
                image_height, image_width, _ = img.shape
                tree = ET.parse(label_path + file)
                for i in tree.iter('object'):
                    xmin = i.find('bndbox/xmin').text
                    ymin = i.find('bndbox/ymin').text
                    xmax = i.find('bndbox/xmax').text
                    ymax = i.find('bndbox/ymax').text
                    i.find('bndbox/xmin').text = str((int(xmin - 15) * width) / image_width)
                    i.find('bndbox/ymin').text = str((int(ymin - 15) * height) / image_height)
                    i.find('bndbox/xmax').text = str((int(xmax + 15) * width) / image_width)
                    i.find('bndbox/ymax').text = str((int(ymax + 15) * height) / image_height)
                dim = (width, height)
                resized = cv2.resize(img, dim)
                cv2.imwrite(image_save_path + file[:-3] + 'jpg', resized)
                tree.write(xml_save_path + file)


# print(get_label("/home/atis/Desktop/Black_Only/DacColor"))

# labeler(path="/home/atis/Dataset/NewDataset/ImageDataset_h5_noLight/Color/",
#         weights_path='/home/atis/Atis/StateMachine/Stonewall/revision/New_AUS_Vision/atis_NonObject2.h5',
#         config_path='/home/atis/Atis/StateMachine/Stonewall/revision/New_AUS_Vision/config_noobj.json')


resize_image_label('/home/atis/Ros/Shovel/ImageDataset/Images_orginal',
                   '/home/atis/Ros/Shovel/ImageDataset/Xml_original',
                   640, 480,
                   xml_save_path='/home/atis/Ros/Shovel/ImageDataset/Xml_416x416/',
                   image_save_path='/home/atis/Ros/Shovel/ImageDataset/Images_416x416/')
