#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import numpy as np

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo
from coordinate_system_conversion.msg import  Coordinate_point,Coordinate_points
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes

class Coordinate_Point:
    def __init__(self) :

        coordinate_information_topic = rospy.get_param(
            '~coordinate_information_topic', '/yolov5/BoundingBoxes')
        camera_info_topic = rospy.get_param(
            '~camera_info_topic', '/camera/color/camera_info')
        pub_topic = rospy.get_param('~pub_topic', '/coordinate_system_conversion/CoordinatePoint')

        self.Boundingboxes = BoundingBoxes()
        self.caminfo = CameraInfo()
        self.K = np.eye(3)

        #subscribe
        self.Boundingbox_sub = rospy.Subscriber(coordinate_information_topic, BoundingBoxes, 
                                          queue_size=10, callback=self.conversion_callback)
        self.caminfo_sub = rospy.Subscriber(camera_info_topic, CameraInfo, 
                                          queue_size=10, callback=self.info_callback)
        
        
        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic, Coordinate_points, queue_size=10)

    def conversion_callback(self, Boundingboxes):
        self.coordinate_points = Coordinate_points()
        self.coordinate_points.header = Boundingboxes.header
        for box in Boundingboxes.bounding_boxes:
            coordinate_point = Coordinate_point()
            coordinate_point.Class = box.Class
            coordinate_point.probability = box.probability
            point = np.array([(box.xmax-box.xmin)/2.0, (box.ymax-box.ymin)/2.0, 1])
            coordinate_point.x, coordinate_point.y, coordinate_point.z = box.distance *np.linalg.inv(self.K) @ point
            self.coordinate_points.coordinate_points.append( coordinate_point)
        print(self.coordinate_points)
        self.position_pub.publish(self.coordinate_points)

    def info_callback(self, caminfo):
        a =  np.array(caminfo.K)
        self.K = a.reshape((3,3))

def main():
    rospy.init_node('yolo_ros', anonymous=True)
    yolo_dect = Coordinate_Point()
    rospy.spin()


if __name__ == "__main__":

    main()