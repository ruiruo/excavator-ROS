#!/usr/bin/env python3.7
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
from coordinate_system_conversion.msg import  Calibrate_flag


class Yolo_Dect:
    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')

        weight_path = rospy.get_param('~weight_path', '')
        color_image_topic = rospy.get_param(
            '~color_image_topic', '/camera/color/image_raw')
        depth_image_topic = rospy.get_param(
            '~depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        calibrate_flag = rospy.get_param('~calibrate_flag', '/coordinate_system_conversion/CalibrateFlag')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        conf = rospy.get_param('~conf', '0.5')

        # load local repository
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.model.cpu()
        else:
            self.model.cuda()
            if (rospy.get_param('/use_half', 'true')):
                self.model.half()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageSec = rospy.get_rostime()
        self.calibrate_flag = Calibrate_flag()

        # image subscribe
        self.calibrate_flag_sub = rospy.Subscriber(calibrate_flag, Calibrate_flag, 
                                          queue_size=10, callback= self.flag_callback)
        self.color_sub = message_filters.Subscriber(color_image_topic, Image, 
                                          queue_size=1, buff_size=9216000)
        self.depth_sub = message_filters.Subscriber(depth_image_topic, Image,
                                          queue_size=1, buff_size=9216000)
        self.ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.image_callback)

        # output publishers
        self.position_pub = rospy.Publisher(pub_topic,  BoundingBoxes, queue_size=10)

        while (not rospy.is_shutdown()) :
            if(rospy.get_rostime().secs-self.getImageSec.secs > 5):
                rospy.logwarn_throttle_identical(5,"waiting for image form target detect node.")
            try:
                if ((self.color_image.shape[0] == 640) and (self.color_image.shape[1] == 640)):
                    cv2.imshow('YOLOv5', self.color_image)
            except:
                pass
            cv2.waitKey(1)

    def flag_callback(self,calibrate_flag):
        self.calibrate_flag.calibrateflag = calibrate_flag.calibrateflag
        if(self.calibrate_flag.calibrateflag == True):
            self.calibrate_flag_sub.unregister()

    def image_callback(self, color_image, depth_image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = color_image.header
        self.boundingBoxes.image_header = color_image.header
        self.getImageSec = rospy.get_rostime()
        rospy.loginfo_throttle_identical(60,"Get image!")
        self.color_image = np.frombuffer(color_image.data, dtype=np.uint8).reshape(
            color_image.height, color_image.width, -1)
        self.depth_image = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(
       color_image.height, color_image.width)
        
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.color_image = cv2.rotate(self.color_image, cv2.ROTATE_90_CLOCKWISE)
        self.color_image = cv2.resize(self.color_image, (640, 640), interpolation=cv2.INTER_LINEAR)
        # self.yuv_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2YUV)
        # self.yuv_image[:,:,0] = cv2.equalizeHist(self.yuv_image[:,:,0])
        # self.color_image = cv2.cvtColor(self.yuv_image, cv2.COLOR_YUV2RGB)
        self.depth_image = cv2.rotate(self.depth_image, cv2.ROTATE_90_CLOCKWISE)

        results = self.model(self.color_image, size=640,augment=False)
        # xmin    ymin    xmax   ymax  confidence  class    name

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

    def get_mid_pos(self, box, depth_data, step):
        distance_list = []
        mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] 
        min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) 
        bias = min_val if min_val < step else step
        scope = int(bias**2//1)
        for i in range(scope):
            row = int(mid_pos[1] + i - bias*(i//bias)-bias//2)
            col = int(mid_pos[0] + i//bias-bias//2)
            dist = depth_data[row, col]
            if dist:
                distance_list.append(dist)
        distance_list = np.array(distance_list)
        distance_list = np.sort(distance_list)[ scope//2- scope//4: scope//2+ scope//4] #冒泡排序+中值滤波
        return np.round(np.mean(distance_list)/10,4) 
    
    def calibrate_yaw(self, box, depth_data):
        if(box[-1] != "Calibration-surface"): return


    def dect(self, boxs):
        for box in boxs:
            #if(not self.calibrate_flag.calibrateflag):
            if(0):
                self.calibrate_yaw(box, self.depth_image)
            else:
                #distance = self.get_randnum_pos(box, self.depth_image, 10)
                boundingBox = BoundingBox()
                boundingBox.probability = np.float64(box[4])
                #boundingBox.distance = np.float64(distance)
                boundingBox.xmin = np.int64(box[0])
                boundingBox.ymin = np.int64(box[1])
                boundingBox.xmax = np.int64(box[2])
                boundingBox.ymax = np.int64(box[3])
                boundingBox.Class = box[-1]
                cv2.rectangle(self.color_image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), [255,255,0], 10)
                self.boundingBoxes.bounding_boxes.append(boundingBox)

        self.position_pub.publish(self.boundingBoxes)

def main():
    rospy.init_node('yolo_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()
