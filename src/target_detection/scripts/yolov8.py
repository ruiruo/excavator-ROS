#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from geometry2.tf2_ros.src import tf2_ros
import numpy as np
from ultralytics import YOLO

import rospy
import message_filters, std_msgs
from sensor_msgs.msg import Image
from target_detection.msg import Box_Point, Target_Points, Key_Point


class TargetDetection:
    def __init__(self):

        # 加载参数
        weight_path = rospy.get_param('~weight_path', 
                                      "$(find target_detection)/weights/yolov8n-pose.onnx")
        color_image_topic = rospy.get_param(
             '~color_image_topic', '/camera/color/image_raw')
        depth_image_topic = rospy.get_param(
            '~depth_image_topic', '/camera/aligned_depth_to_color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/target_detection/Target_Points')
        # function rospy.get_param spend about 5ms on TX2
        self.half = rospy.get_param('~half', 'False')
        self.conf = rospy.get_param('~conf', '0.5')
        self.point_search_range = rospy.get_param('~point_search_range', '4')
        self.cal_search_num = int(rospy.get_param('~search_num', '4'))+1
        self.model = YOLO(weight_path, task='pose')

        # 选择yolo运行设备与是否使用半浮点
        if (rospy.get_param('~cpu', 'False')):
            self.device = 'cpu'
        else:
            self.device = '0'

        #realsense相机图像话题订阅
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageSec = rospy.get_rostime()
        self.color_sub = message_filters.Subscriber(color_image_topic, Image, 
                                          queue_size=1, buff_size=9216000)
        self.depth_sub = message_filters.Subscriber(depth_image_topic, Image,
                                          queue_size=1, buff_size=9216000)
        self.ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.image_callback)

        self.calibrate_buffer = tf2_ros.Buffer(rospy.Time())
        self.calibrate_listener = tf2_ros.TransformListener(self.calibrate_buffer)

        #发布检测框与关键点话题
        self.position_pub = rospy.Publisher(pub_topic, Target_Points, queue_size=10)

        while (not rospy.is_shutdown()) :
            if(rospy.get_rostime().secs-self.getImageSec.secs > 5):
                rospy.logwarn_throttle_identical(20,"Waiting for image form target detect node.")
            try:
                    cv2.imshow("yolov8 inf",self.image)
                    cv2.waitKey(1)
            except Exception as e:
                pass

    def image_callback(self, color_image, depth_image):
        #获取相机话题信息并通知
        self.TargetPoints = Target_Points()
        self.TargetPoints.header = color_image.header
        self.getImageSec = rospy.get_rostime()
        rospy.loginfo_throttle_identical(600,"Get image!")

        #对RGBD相机信息整理与与处理
        self.color_image = np.frombuffer(color_image.data, dtype=np.uint8).reshape(
            color_image.height, color_image.width, -1)
        self.depth_image = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(
       depth_image.height, depth_image.width)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        # Rotation spend about 8-11ms on TX2
        self.color_image = cv2.rotate(self.color_image, cv2.ROTATE_90_CLOCKWISE)
        # equalizeHist spend about 7-11ms on TX2
        # yuv_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2YUV)
        # yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
        # self.color_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
        self.depth_image = cv2.rotate(self.depth_image, cv2.ROTATE_90_CLOCKWISE)

        #输出模型监测结果，并导出预览图片与整理检测结果
        # yolov8-pose (half) spend about 100-125ms on TX2
        results = self.model(self.color_image, conf=self.conf, 
                             device=self.device, half=self.half)
        # yolov8-pose plot spend about 25-35ms on TX2
        self.image = results[0].plot()

        if any(results[0].boxes.cls):
            boxes = [
                [results[0].names[int(cls)]] + [float(conf)] + xywh.tolist() + keypoints_xy.tolist()
                for cls, conf, xywh, keypoints_xy in zip(
                    results[0].boxes.cls,
                    results[0].boxes.conf,
                    results[0].boxes.xywh,
                    results[0].keypoints.xy)]
            self._data_push(boxes)#spend 2ms
        else:
           rospy.logwarn_throttle_identical(10,"No valid targets detected")

    def _data_push(self, boxes):
        can_transform = self.calibrate_buffer.can_transform("camera_color_frame", "body", rospy.Time())
        for box in boxes:
            if not can_transform and box[0] == "Calibration_surface":
                self._collect_boxes(box[:6])
            elif can_transform:
                self._collect_boxes(box[:6])
                self._collect_keypoints(box[6:])
        self.position_pub.publish(self.TargetPoints)

    def _collect_boxes(self, box):
        BoxPoint = Box_Point(
            probability=box[1],
            distance=self._get_forearm_attitude(box),
            x=box[2], y=box[3],
            width=box[4],  height=box[5],
            name=box[0]
        )
        self.TargetPoints.box_points.append(BoxPoint)

    def _collect_keypoints(self, keypoints):
        kp_name = ['A', 'B', 'A_d', 'B_d', 'BF_l', 'BF_r']
        self.TargetPoints.key_points.extend(
            Key_Point(
                x=kp[0], y=kp[1],
                distance=self._get_point_distance([int(kp[0]), int(kp[1])]),
                name=name
            ) for name, kp in zip(kp_name, keypoints) if self._get_point_distance([int(kp[0]), int(kp[1])]) != 0.
        )

    def _get_point_distance(self, mid_pos):
        for scope in range(int(self.point_search_range)):
            if (mid_pos[0] + scope < self.depth_image.shape[1] and 
                mid_pos[1] + scope < self.depth_image.shape[0]):
                distance_list = np.array([self.depth_image[int(mid_pos[1]+i)][int(mid_pos[0]+j)]
                                        for i in range(-scope, scope+1) for j in range(-scope, scope+1)])
            if np.sum(distance_list) != 0 :
                break
        distance_list = distance_list[distance_list != 0]
        if len(distance_list)>=4:
            distance_list = np.percentile(distance_list, [25, 50, 75])
        elif len(distance_list) == 0:
            return 0. 
        return np.round(np.mean(distance_list) / 10, 4)#距离单位cm 原始数据单位为mm

    def _get_forearm_attitude(self, box):
        if box[0] != "Calibration_surface":
            return 0.
        mid_pos = [int(box[2]), int(box[3])]
        mid_distance = self._get_point_distance(mid_pos)
        search_dis = box[4]/4
        search_offsets = [0, -search_dis, search_dis]
        search_list = [(mid_pos[0] + dx, mid_pos[1] + dy)
                    for dx in search_offsets for dy in search_offsets]
        
        for i, (x, y) in enumerate(search_list):
            if i >= self.cal_search_num:
                break 
            distance = self._get_point_distance([x, y])
            if distance != 0:
                KeyPoint = Key_Point(x=x, y=y, distance=distance,
                                    name=f"Calibration_surface_{i}")
                self.TargetPoints.key_points.append(KeyPoint)
        return mid_distance
        

 
def main():
    rospy.init_node('target_detection', anonymous=True)
    yolo_dect = TargetDetection()
    rospy.spin()


if __name__ == "__main__":
    main()
