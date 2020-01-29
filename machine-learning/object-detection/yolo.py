import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet

# Set the NMS Threshold
score_threshold = 0.6
# Set the IoU threshold
iou_threshold = 0.4
cfg_file = "cfg/yolov3.cfg"
weight_file = "weights/yolov3.weights"
namesfile = "data/coco.names"
m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
# m.print_network()
original_image = cv2.imread("images/city_scene.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
img = cv2.resize(original_image, (m.width, m.height))
# detect the objects
boxes = detect_objects(m, img, iou_threshold, score_threshold)
# plot the image with the bounding boxes and corresponding object class labels
plot_boxes(original_image, boxes, class_names, plot_labels=True)