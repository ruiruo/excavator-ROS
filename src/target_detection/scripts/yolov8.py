#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
import random
from ultralytics import YOLO

import message_filters
from sensor_msgs.msg import Image
from target_detection.msg import Target_Point, Target_Points
from coordinate_system_conversion.msg import  Calibrate_flag


class TargetDetection:
    def __init__(self):

        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        color_image_topic = rospy.get_param(
            '~color_image_topic', '/camera/color/image_raw')
        depth_image_topic = rospy.get_param(
            '~depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        calibrate_flag = rospy.get_param('~calibrate_flag', '/coordinate_system_conversion/CalibrateFlag')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/Target_Points')
        self.conf = rospy.get_param('~conf', '0.5')

        self.model = YOLO(weight_path, task='pose')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'false')):
            self.device = 'cpu'
        else:
            self.device = '0'
            if (rospy.get_param('/use_half', 'false')):
                self.half = True

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
        self.position_pub = rospy.Publisher(pub_topic, Target_Points, queue_size=10)

        while (not rospy.is_shutdown()) :
            if(rospy.get_rostime().secs-self.getImageSec.secs > 5):
                rospy.logwarn_throttle_identical(5,"waiting for image form target detect node.")
            try:
                cv2.imshow("YOLOv8 Inference", self.annotated_frame)
            except:
                pass
            cv2.waitKey(1)

    def flag_callback(self,calibrate_flag):
        self.calibrate_flag.calibrateflag = calibrate_flag.calibrateflag
        if(self.calibrate_flag.calibrateflag == True):
            self.calibrate_flag_sub.unregister()

    def image_callback(self, color_image, depth_image):
        self.TargetPoints = Target_Points()
        self.TargetPoints.header = color_image.header
        self.getImageSec = rospy.get_rostime()
        rospy.loginfo_throttle_identical(60,"Get image!")
        self.color_image = np.frombuffer(color_image.data, dtype=np.uint8).reshape(
            color_image.height, color_image.width, -1)
        self.depth_image = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(
       depth_image.height, depth_image.width)
        
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.color_image = cv2.rotate(self.color_image, cv2.ROTATE_90_CLOCKWISE)
        self.color_image = cv2.resize(self.color_image, (640, 640), interpolation=cv2.INTER_LINEAR)
        self.yuv_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2YUV)
        self.yuv_image[:,:,0] = cv2.equalizeHist(self.yuv_image[:,:,0])
        self.color_image = cv2.cvtColor(self.yuv_image, cv2.COLOR_YUV2RGB)
        self.depth_image = cv2.rotate(self.depth_image, cv2.ROTATE_90_CLOCKWISE)
        print( self.depth_image.shape)

        results = self.model(self.color_image, conf=self.conf, device=self.device, half=self.half)
        # xmin    ymin    xmax   ymax  confidence  class    name
        self.annotated_frame = results[0].plot(labels=False, conf=True)

        boxs = results.pandas().xywh[0].values
        print(boxs)
        # self.dect( boxs)

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
        distance_list = np.sort(distance_list)[ scope//2- scope//4: scope//2+ scope//4] 
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
                TargetPoint = Target_Point()
                TargetPoint.probability = np.float64(box[4])
                #TargetPoint.distance = np.float64(distance)
                TargetPoint.x = np.int64(box[0])
                TargetPoint.y = np.int64(box[1])
                TargetPoint.width = np.int64(box[2])
                TargetPoint.height = np.int64(box[3])
                TargetPoint.Class = box[-1]
                cv2.rectangle(self.color_image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), [255,255,0], 10)
                self.TargetPoints.target_points.append(boundingBox)

        self.position_pub.publish(self.boundingBoxes)

def main():
    rospy.init_node('target_detection', anonymous=True)
    yolo_dect = TargetDetection()
    rospy.spin()


if __name__ == "__main__":
    main()