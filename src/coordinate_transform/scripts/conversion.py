#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from itertools import combinations
from scipy.spatial.transform import Rotation as R
import scipy.constants as C

from geometry2.tf2_ros.src import tf2_ros
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo,Imu
from coordinate_transform.msg import   Coordinate_point, Coordinate_points
from target_detection.msg import Target_Points

class Coordinate_Point:
    def __init__(self) :
        #获取参数与变量初始化
        coordinate_information_topic = rospy.get_param(
            '~coordinate_information_topic', '/target_detection/Target_Points')
        camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')
        accel_info_topic = rospy.get_param('~accel_info_topic', '/camera/accel/sample')

        pub_topic = rospy.get_param('~pub_topic', '/coordinate_transform/CoordinatePoint')

        self.TargetPoints = Target_Points()
        self.cam_info = CameraInfo()
        self.accel_info = Imu()
        self. color_transform = TransformStamped()
        self.accel_transform = TransformStamped()
        self.cam_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.color_cam_to_body_tf = TransformStamped()
        self.yaw_list = []

        self.get_sub_time = [rospy.get_rostime(),rospy.get_rostime()]
        self.K = np.eye(3)
        self.calibration_level_matrix = np.eye(3)
        self.calibrate_array = np.empty((3,0),dtype=np.float64)

        #相机内参获取定时器
        self.cam_info_rate = rospy.Rate(1.)

        #话题订阅
        self.Target_sub = rospy.Subscriber(coordinate_information_topic, Target_Points, 
                                          queue_size=10, callback=self.conversion_callback)
        self.cam_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, 
                                          queue_size=10, callback=self.info_callback)
        self.accel_info_sub = rospy.Subscriber(accel_info_topic, Imu, 
                                               queue_size=5, callback=self.calibration_level_callback)
        self.conversion_buffer = tf2_ros.Buffer(rospy.Time())
        self.conversion_listener = tf2_ros.TransformListener(self.conversion_buffer)
        
        # 话题发布
        self.position_pub = rospy.Publisher(pub_topic, Coordinate_points, queue_size=10)

        #订阅话题健康检查
        while(not rospy.is_shutdown() ):
            if(rospy.get_rostime().secs - self.get_sub_time[0].secs > 10 ):
                rospy.logwarn_throttle_identical(60,"Waiting for target detect form yolo node.")
            if(rospy.get_rostime().secs - self.get_sub_time[1].secs > 2 ):
                rospy.logwarn_throttle_identical(20,"Waiting for accel info form realsense node.")

    #将校正后得到的相机外参作用于关键点并发布
    def conversion_callback(self, TargetPoints):
        self.get_sub_time[0] = rospy.get_rostime()
        self.TargetPoints = TargetPoints
        if self.conversion_buffer.can_transform("camera_color_frame", "body",  rospy.Time()):
            self.coordinate_points = Coordinate_points()
            self.coordinate_points.header = TargetPoints.header
            inv_K = np.linalg.inv(self.K) 
            for p in TargetPoints.key_points:
                coordinate_point = Coordinate_point()
                coordinate_point.name = p.name
                point = np.array([[p.x], [p.y], [1]])
                point= np.dot(p.distance * inv_K , point)
                point[[1, 2]] = point[[2, 1]]
                #x,y,z 下 前 左
                coordinate_point.x, coordinate_point.y, coordinate_point.z = np.dot(
                    point.T,self.R_color_to_body.as_matrix()).T[[1, 2, 0]]
                coordinate_point.y = -coordinate_point.y
                #x,y,z 前 右 下
                self.coordinate_points.coordinate_points.append( coordinate_point)
            self.position_pub.publish(self.coordinate_points)

    #获取相机内参并将内参按照图片旋转90度变换
    def info_callback(self, cam_info):
        self.cam_info_rate.sleep()
        a =  np.array(cam_info.K)
        a[[0,2,4,5]] = a[[4,5,0,2]]
        self.K = a.reshape(3,3)

    #经校正获取相机内参
    def calibration_level_callback(self, accel_info):
        self.get_sub_time[1] = rospy.get_rostime()
        if(not self.conversion_buffer.can_transform("camera_color_frame", "body",  rospy.Time())):
            rospy.loginfo_throttle_identical(120,
                                             "Calibrating...\n Do not move the camera and the devices rigidly connected to it!!!")
        
            #获取加速度计数值
            self._collect_accel_info(accel_info)
            get_yaw = self._get_yaw()

            #满足校准条件开始校准
            if((np.size(self.calibrate_array) >= 64) and get_yaw and
                (not self.conversion_buffer.can_transform("camera_color_frame", "body", rospy.Time()))):
                #获取坐标系转换矩阵信息
                if(self.conversion_buffer.can_transform("camera_link",  "camera_aligned_depth_to_color_frame", 
                                                            rospy.Time())):
                    self.color_transform = self.conversion_buffer.lookup_transform( "camera_link",
                                                                                "camera_color_frame",  rospy.Time())
                    self.accel_transform = self.conversion_buffer.lookup_transform("camera_link", "camera_accel_frame",  
                                                                        rospy.Time())
                    self._calibrate()
                else:
                    rospy.logwarn_throttle_identical(20,"Check if necessary tf exists.")

    #获取加速度计读数
    def _collect_accel_info(self, info):
        #accel_info.linear_acceleration.x,y,z 东 天 北 eun
        xyz = np.array([[info.linear_acceleration.x], [info.linear_acceleration.y], [info.linear_acceleration.z]], 
                       dtype=np.float64)
        self.calibrate_array = np.hstack((self.calibrate_array, xyz))

    #外参旋转矩阵获取
    def _calibrate(self):
        calibrate_angle = np.einsum('ij->j', np.percentile(self.calibrate_array, [25, 50, 75], axis=1)) / 3.
        calibrate_angle = np.clip(calibrate_angle, -C.g, C.g) 
        roll = math.degrees(math.atan(-calibrate_angle[0] /         #使得roll的旋转方向为绕前顺时针旋转为正
                                        math.sqrt(calibrate_angle[1]**2 + calibrate_angle[2]**2)
                                        if calibrate_angle[1]**2 + calibrate_angle[2]**2 != 0. else float('inf')))
        pitch =  90. + math.degrees(math.atan(calibrate_angle[1] / 
                                                math.sqrt(calibrate_angle[0]**2 + calibrate_angle[2]**2) 
                                                if calibrate_angle[0]**2 + calibrate_angle[2]**2 != 0. else float('inf')))
        yaw = -np.einsum('i->', np.percentile(self.yaw_list, [25, 50, 75])) / 3.
        self.calibrate_array = np.empty((3,0),dtype=np.float64)
        self.yaw_list = []
        rospy.loginfo("Calibration successful! ")
        rospy.loginfo("pitch = %2f  roll = %2f yaw = %2f", pitch, roll, yaw)
        self._color_to_body_coordinate_system(pitch,roll, yaw)

    #通过小臂校正面获取彩色相机和集体的yaw偏差
    def _get_yaw(self):
        inv_K = np.linalg.inv(self.K)
        calibration_reference = rospy.get_param('~calibration_reference', '100')
        point_list = [np.dot(point.distance * inv_K, np.hstack([point.x, point.y, 1])) 
                    for point in self.TargetPoints.key_points 
                    if point.name.startswith("Calibration_surface") and
                    point.distance <= calibration_reference]

        if np.sum(point_list) > 50000 or len(point_list) < 3:
            return False

        point_array = np.array(point_list)
        surface_list = list(combinations(point_array, 3))
        yaw_vector_list = [np.cross(surface[1] - surface[0], surface[2] - surface[0]) 
                        for surface in surface_list]
        yaw_vector_list = [v / np.linalg.norm(v) for v in yaw_vector_list if np.linalg.norm(v) > 0]

        norm_z = np.linalg.norm([0, 0, 1])
        for vector in yaw_vector_list:
            cos_angle = np.dot(vector, [0, 0, 1]) / (np.linalg.norm(vector) * norm_z)
            yaw = np.degrees(np.arccos(cos_angle))
            self.yaw_list.append(yaw)
        return True

    def _color_to_body_coordinate_system(self, pitch, roll, yaw):
        R_body_to_color_yaw = R.from_euler('XYZ', [0, 0, yaw], degrees=True)
        R_camera_to_color = R.from_quat([self.color_transform.transform.rotation.x, 
                              self.color_transform.transform.rotation.y, 
                              self.color_transform.transform.rotation.z, 
                              self.color_transform.transform.rotation.w])
        R_body_to_camera_yaw = R_body_to_color_yaw * R_camera_to_color 
        # 加速度计坐标到相机坐标系仅有平移关系
        R_world_to_accel = R.from_euler('XYZ', [pitch, roll, 0], degrees=True)
        #缺少body与world的旋转矩阵 yaw直接参考机体 roll与pitch参考世界坐标系
        self.R_color_to_body = ((R_world_to_accel * R_body_to_camera_yaw).inv())
        # print(self.R_color_to_body.as_euler('XYZ', degrees=True))
        self.color_cam_to_body_tf.header.frame_id = "camera_color_frame"
        self.color_cam_to_body_tf.header.stamp = rospy.Time.now()
        self.color_cam_to_body_tf.header.seq = 0
        self.color_cam_to_body_tf.child_frame_id = "body"
        self.color_cam_to_body_tf.transform.translation.x = 0.
        self.color_cam_to_body_tf.transform.translation.y = 0.
        self.color_cam_to_body_tf.transform.translation.z = 0.
        self.color_cam_to_body_tf.transform.rotation.x = self.R_color_to_body.as_quat()[0]
        self.color_cam_to_body_tf.transform.rotation.y = self.R_color_to_body.as_quat()[1]
        self.color_cam_to_body_tf.transform.rotation.z = self.R_color_to_body.as_quat()[2]
        self.color_cam_to_body_tf.transform.rotation.w = self.R_color_to_body.as_quat()[3]
        self.cam_broadcaster.sendTransform(self.color_cam_to_body_tf)


def main():
    rospy.init_node('coordinate_transform', anonymous=True)
    coordinate_transform = Coordinate_Point()
    rospy.spin()


if __name__ == "__main__":

    main()
