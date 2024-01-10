#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

import rospy, tf2_ros
import pytransform3d as pt
from geometry_msgs.msg import TransformStamped, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo,Imu
from coordinate_system_conversion.msg import  Calibrate_flag, Coordinate_point, Coordinate_points
from target_detection.msg import Target_Point, Target_Points

class Coordinate_Point:
    def __init__(self) :

        coordinate_information_topic = rospy.get_param(
            '~coordinate_information_topic', '/target_detection/Target_Points')
        camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')
        accel_info_topic = rospy.get_param('~accel_info_topic', '/camera/accel/sample')

        pub_topic = rospy.get_param('~pub_topic', '/coordinate_system_conversion/CoordinatePoint')
        pub_flag_topic = rospy.get_param('~pub_flag_topic', '/coordinate_system_conversion/CalibrateFlag')

        self.TargetPoints = Target_Points()
        self.cam_info = CameraInfo()
        self.accel_info = Imu()
        self.K = np.eye(3)
        self.calibration_level_matrix = np.eye(3)
        self.calibrate_array = np.array([0,0,0],dtype=float)
        self.calibrate_count = 0
        self.calibrate_flag = Calibrate_flag()
        self.calibrate_flag.calibrateflag = False
        self. color_transform = TransformStamped()
        self.accel_transform = TransformStamped()
        self.get_sub_time = [rospy.get_rostime(),rospy.get_rostime()]

        #timer
        self.cam_info_rate = rospy.Rate(1.)
        self.flag_timer = rospy.Timer(rospy.Duration(1),callback=self.flag_callback)

        #subscribers
        self.Boundingbox_sub = rospy.Subscriber(coordinate_information_topic, Target_Points, 
                                          queue_size=10, callback=self.conversion_callback)
        self.cam_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, 
                                          queue_size=10, callback=self.info_callback)
        self.accel_info_sub = rospy.Subscriber(accel_info_topic, Imu, 
                                               queue_size=5, callback=self.calibration_level_callback)
        self.conversion_buffer = tf2_ros.Buffer(rospy.Time())
        self.conversion_listener = tf2_ros.TransformListener(self.conversion_buffer)
        
        # output publishers
        self.position_pub = rospy.Publisher(pub_topic, Coordinate_points, queue_size=10)
        self.flag_pub = rospy.Publisher(pub_flag_topic, Calibrate_flag, queue_size=10)

        while(not rospy.is_shutdown() ):
            if(rospy.get_rostime().secs - self.get_sub_time[0].secs > 5 ):
                rospy.logwarn_throttle_identical(5,"waiting for target detect form yolo node.")
            if(rospy.get_rostime().secs - self.get_sub_time[1].secs > 2 ):
                rospy.logwarn_throttle_identical(5,"waiting for accel info form realsense node.")

    def flag_callback(self, event):
        if(self.calibrate_flag.calibrateflag==True):
            self.flag_pub.unregister()
        else:
            self.flag_pub.publish(self.calibrate_flag)

    def conversion_callback(self, TargetPoints):
        self.get_sub_time[0] = rospy.get_rostime()
        self.coordinate_points = Coordinate_points()
        self.coordinate_points.header = TargetPoints.header
        for box in TargetPoints.target_points:
            if((box.distance == np.nan)): continue
            coordinate_point = Coordinate_point()
            coordinate_point.Class = box.Class
            coordinate_point.probability = box.probability
            point = np.array([(box.xmax-box.xmin)/2.0, (box.ymax-box.ymin)/2.0, 1])
            #x,y,z 东 天 北 eun
            coordinate_point.x, coordinate_point.y, coordinate_point.z = np.dot(box.distance *np.linalg.inv(self.K) , point)
            #缺少仿射矩阵从相机坐标转换至机体坐标
            self.coordinate_points.coordinate_points.append( coordinate_point)
        self.position_pub.publish(self.coordinate_points)

    def info_callback(self, cam_info):
        self.cam_info_rate.sleep()
        a =  np.array(cam_info.K)
        self.K = a.reshape((3,3))

    def calibration_level_callback(self, accel_info):#accel_info.linear_acceleration.x,y,z 东 天 北 eun
        self.get_sub_time[1] = rospy.get_rostime()
        if(not self.calibrate_flag.calibrateflag):
            rospy.loginfo_throttle_identical(3,
                                             "Calibrating...\n Do not move the camera and the devices rigidly connected to it!!!")
            
        #获取加速度计数值
        xyz = [accel_info.linear_acceleration.x, accel_info.linear_acceleration.y, accel_info.linear_acceleration.z]
        if(sum(xyz) != 0):
            for i in range(3):
                self.calibrate_array[i] = (self.calibrate_array[i] + xyz[i])/2.0
            self.calibrate_count += 1

        #获取坐标系转换矩阵信息
        if(self.conversion_buffer.can_transform("camera_link",  "camera_aligned_depth_to_color_frame", 
                                                    rospy.Time())):
            self.color_transform = self.conversion_buffer.lookup_transform( "camera_link",
                                                                           "camera_color_frame",  rospy.Time())
            self.accel_transform = self.conversion_buffer.lookup_transform("camera_link", "camera_accel_frame",  
                                                                rospy.Time())
        
        #满足校准条件开始校准
        if((self.calibrate_count >= 70) and (not self.calibrate_flag.calibrateflag) and 
           (( self.accel_transform.transform.translation.x != 0.) or 
            (self.color_transform.transform.rotation.x != 0.))):
            self.calibrate_count = 0
            g = 9.5 
            self.calibrate_array[0] = max(min(-self.calibrate_array[0], g), -g) 
            self.calibrate_array[1] = max(min(self.calibrate_array[1], g), -g)
            self.calibrate_array[2] = max(min(self.calibrate_array[2], g), -g) 
            roll = math.degrees(math.atan(self.calibrate_array[0] / math.sqrt(self.calibrate_array[1]**2 +
                                                                          self.calibrate_array[2]**2)) )
            pitch = 90. + math.degrees(math.atan(self.calibrate_array[1] / math.sqrt(self.calibrate_array[0]**2 +
                                                                          self.calibrate_array[2]**2)) )
            self.calibrate_array = [0,0,0]
            self.color_to_world_coordinate_system(pitch,roll)
            rospy.loginfo_once("Calibration successful! \n pitch = %f  roll = %f", pitch, roll)
            self.calibrate_flag.calibrateflag = True

        
    def color_to_world_coordinate_system(self, pitch, roll):
        R_camera_color = pt.rotations.quaternion_matrix([self.color_transform.transform.rotation.w, 
                                                 self.color_transform.transform.rotation.x, 
                                                 self.color_transform.transform.rotation.y, 
                                                 self.color_transform.transform.rotation.z])[:3, :3]

        # 加速度计到相机坐标系仅有平移关系
        # 缺少视觉相机获取yaw
        # 将欧拉角转换为旋转矩阵
        R_world_accel = pt.rotations.euler_zyx_matrix([0, np.deg2rad(pitch), np.deg2rad(roll)])[:3, :3]
        R_color_world = np.dot(R_camera_color.T, R_world_accel.T)


def main():
    rospy.init_node('coordinate_system_conversion', anonymous=True)
    coordinate_system_conversion = Coordinate_Point()
    rospy.spin()


if __name__ == "__main__":

    main()
