#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
import random

import message_filters
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes


class Yolo_Dect:
    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')

        weight_path = rospy.get_param('~weight_path', '')
        color_image_topic = rospy.get_param(
            '~color_image_topic', '/camera/color/image_raw')
        depth_image_topic = rospy.get_param(
            '~depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        conf = rospy.get_param('~conf', '0.5')

        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False

        # image subscribe
        self.color_sub = message_filters.Subscriber(color_image_topic, Image, 
                                          queue_size=1, buff_size=52428800)
        self.depth_sub = message_filters.Subscriber(depth_image_topic, Image,
                                          queue_size=1, buff_size=52428800)
        self.ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.image_callback)

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=10)

        while (not self.getImageStatus) :
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def image_callback(self, color_image, depth_image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = color_image.header
        self.boundingBoxes.image_header = color_image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(color_image.data, dtype=np.uint8).reshape(
            color_image.height, color_image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image)
        # xmin    ymin    xmax   ymax  confidence  class    name
        self.depth_data = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(
       color_image.height, color_image.width)

        boxs = results.pandas().xyxy[0].values
        self.dect( boxs)

    def get_randnum_pos(self, box,depth_data,randnum):
        distance_list = []
        mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
        min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
        for i in range(randnum):
            bias = random.randint(-min_val//4, min_val//4)
            dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
            if dist:
                distance_list.append(dist)
        distance_list = np.array(distance_list)
        distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
        return np.round(np.mean(distance_list)/10,4) #距离单位为cm

    def get_mid_pos(self, box,depth_data, step):
        distance_list = []
        mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] 
        min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) 
        bias = min_val if min_val < step else step
        scope = bias**2
        for i in range( scope):
            dist = depth_data[int(mid_pos[1] + i - bias*(i//bias)), int(mid_pos[0] + i//bias)]
            if dist:
                distance_list.append(dist)
        distance_list = np.array(distance_list)
        distance_list = np.sort(distance_list)[ scope//2- scope//4: scope//2+ scope//4] #冒泡排序+中值滤波
        return np.round(np.mean(distance_list)/10,4) 

    def dect(self, boxs):

        for box in boxs:
            distance = self.get_mid_pos(box, self.depth_data, 10)
            boundingBox = BoundingBox()
            boundingBox.probability = np.float64(box[4])
            boundingBox.distance = np.float64(distance)
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.Class = box[-1]

            self.boundingBoxes.bounding_boxes.append(boundingBox)
            self.position_pub.publish(self.boundingBoxes)

def main():
    rospy.init_node('yolo_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()
