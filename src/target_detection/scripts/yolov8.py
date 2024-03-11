#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import math
from itertools import combinations
from scipy.spatial.transform import Rotation as R
import scipy.constants as C

import rospy
from geometry2.tf2_ros.src import tf2_ros
from geometry_msgs.msg import TransformStamped
from target_detection.msg import Box_Point, Target_Points, Key_Point, Coordinate_point, Coordinate_points


class TargetDetection:
    def __init__(self):

        # 加载参数
        weight_path = rospy.get_param('~weight_path', 
                                      "$(find target_detection)/weights/yolov8n-pose.onnx")
        pub_topic = rospy.get_param('~pub_topic', '/target_detection/Target_Points')
        self.calibration_reference = rospy.get_param('~calibration_reference', '100')

        pub_topic = rospy.get_param('~pub_topic', '/coordinate_transform/CoordinatePoint')
        # function rospy.get_param spend about 5ms on TX2
        self.half = rospy.get_param('~half', 'False')
        self.show = rospy.get_param('~show', 'False')
        self.conf = rospy.get_param('~conf', '0.5')
        self.point_search_range = rospy.get_param('~point_search_range', '4')
        self.cal_search_num = int(rospy.get_param('~search_num', '4'))+1
        self.model = YOLO(weight_path, task='pose')
        self.debug = rospy.get_param('~debug', 'False')

        # 选择yolo运行设备
        if (rospy.get_param('~cpu', 'False')):
            self.device = 'cpu'
        else:
            self.device = '0'

        self.calibrate_buffer = tf2_ros.Buffer(rospy.Time())
        self.calibrate_listener = tf2_ros.TransformListener(self.calibrate_buffer)
        self.TargetPoints = Target_Points()
        self.cam_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.color_cam_to_body_tf = TransformStamped()
        self.yaw = []
        self.seq = 0
        # 彩色相机内参按照图片旋转90度变换
        self.K = np.array([[608.38751221, 0., 233.09431458], 
                           [  0., 608.90325928, 411.84860229],[  0., 0., 1., ]], dtype=np.float64)
        self.calibration_level_matrix = np.eye(3)
        self.calibrate_array = np.empty((3,0), dtype=np.float64)

        #相机图像pipeline配置
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        self.conversion_buffer = tf2_ros.Buffer(rospy.Time())
        self.conversion_listener = tf2_ros.TransformListener(self.conversion_buffer)

        #发布检测框与关键点话题
        self.position_pub = rospy.Publisher(pub_topic, Coordinate_points, queue_size=10)

        while (not rospy.is_shutdown()) :
            try:
                    if self.show:
                        cv2.imshow("yolov8 inf", image)
                        cv2.waitKey(1)
            except Exception as e:
                pass
            
            try:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                accel_frame = frames.first_or_default(rs.stream.accel)

                if accel_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    xyz = np.array([[accel_data.x], [accel_data.y], [accel_data.z]], dtype=np.float64)
                    self.calibrate_array = np.hstack((self.calibrate_array, xyz))

                if  aligned_depth_frame and color_frame:
                    getImageSec = rospy.get_rostime()
                    rospy.loginfo_throttle_identical(600,"Get image!")
                    self.TargetPoints = Target_Points()
                    self.coordinate_points = Coordinate_points()
                    self.coordinate_points.header.stamp = rospy.Time.now()
                    self.seq =+ 1
                    self.coordinate_points.header.seq = self.seq
                    self.coordinate_points.header.frame_id = "camera_color_frame"

                    self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
                    self.depth_image = cv2.rotate(self.depth_image, cv2.ROTATE_90_CLOCKWISE)

                    #输出模型监测结果，并导出预览图片与整理检测结果
                    results = self.model(color_image, conf=self.conf, device=self.device, half=self.half)

                    if self.show:
                        image = results[0].plot()

                    if any(results[0].boxes.cls):
                        boxes = [
                            [results[0].names[int(cls)]] + [float(conf)] + xywh.tolist() + keypoints_xy.tolist()
                            for cls, conf, xywh, keypoints_xy in zip(
                                results[0].boxes.cls,
                                results[0].boxes.conf,
                                results[0].boxes.xywh,
                                results[0].keypoints.xy)]
                        self._data_process(boxes)#spend 2ms
                    else:
                        rospy.logwarn_throttle_identical(10,"No valid targets detected")

                get_yaw = self._get_yaw()
                #将校正后得到的相机外参作用于关键点并发布
                if((np.size(self.calibrate_array) >= 250) and get_yaw and
                    (not self.conversion_buffer.can_transform("camera_color_frame", "body", rospy.Time()))):
                    self._calibrate()

                if self.conversion_buffer.can_transform("camera_color_frame", "body",  rospy.Time()):
                    inv_K = np.linalg.inv(self.K) 
                    for p in self.TargetPoints.key_points:
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
                
                if(rospy.get_rostime().secs - getImageSec.secs > 20):
                    rospy.logwarn_throttle_identical(20,"Waiting for image form target detect node.")
            except Exception as e:
                rospy.logwarn_throttle_identical(60,\
                                                 "Unable to obtain camera image or aceel data, check if camera is working.")
        pipeline.stop()
        cv2.destroyAllWindows()

    def _data_process(self, boxes):
        can_transform = self.calibrate_buffer.can_transform("camera_color_frame", "body", rospy.Time())
        for box in boxes:
            if not can_transform and box[0] == "Calibration_surface":
                self._collect_boxes(box[:6])
            elif can_transform:
                self._collect_boxes(box[:6])
                self._collect_keypoints(box[6:])

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
        search_dis = int(box[4]/4)
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
        yaw = np.round(np.mean(self.yaw), 4)
        self.calibrate_array = np.empty((3,0), dtype=np.float64)
        self.yaw_list = []
        rospy.loginfo("Calibration successful! ")
        rospy.loginfo("pitch = %2f  roll = %2f yaw = %2f", pitch, roll, yaw)
        self._color_to_body_coordinate_system(pitch, roll, yaw)

    #通过小臂校正面获取彩色相机和集体的yaw偏差
    def _get_yaw(self):
        def draw_histogram(data):
            from collections import Counter
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from PIL import Image
            def show_plot():
                try:
                    if self.img is not None:
                        self.img.close()
                except Exception as e:
                    print(f'关闭图像时出错: {e}')
                self.img = Image.open('plot.png')
                self.img.show()
            if not all(isinstance(n, int) for n in data):
                raise ValueError('数据列表必须只包含整数')

            plt.clf()

            data_counts = Counter(data)
            top_three = data_counts.most_common(3)

            plt.hist(data, bins=range(min(data), max(data) + 2), align='left', color='blue', edgecolor='black')
            for value, count in top_three:
                plt.text(value, count, f'{value}')
            plt.savefig('plot.png')
            show_plot()

        yaw_list=[]
        inv_K = np.linalg.inv(self.K)
        point_list = [np.dot(point.distance * inv_K, np.hstack([point.x, point.y, 1])) 
                    for point in self.TargetPoints.key_points 
                    if point.name.startswith("Calibration_surface") and
                    point.distance <= self.calibration_reference]

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
            yaw_list.append(int(yaw))
        
        if self.debug:
            draw_histogram(yaw_list)
        
        self.yaw.append(-np.einsum('i->', np.percentile(yaw_list, [25, 50, 75])) / 3.)
        if len(self.yaw) > 3:
            if np.std(self.yaw[-3:]) < 3:  
                return True
        return False

    def _color_to_body_coordinate_system(self, pitch, roll, yaw):
        R_body_to_color_yaw = R.from_euler('XYZ', [0, 0, yaw], degrees=True)
        R_camera_to_color = R.from_quat([0.00121952651534, -0.00375633803196, 
                                         -0.000925257743802, 0.999991774559])
        R_body_to_camera_yaw = R_body_to_color_yaw * R_camera_to_color 
        # 加速度计坐标到相机坐标系仅有平移关系
        R_world_to_accel = R.from_euler('XYZ', [pitch, roll, 0], degrees=True)
        #缺少body与world的旋转矩阵 yaw直接参考机体 roll与pitch参考世界坐标系
        self.R_color_to_body = ((R_world_to_accel * R_body_to_camera_yaw).inv())

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
    rospy.init_node('target_detection', anonymous=True)
    yolo_dect = TargetDetection()
    rospy.spin()


if __name__ == "__main__":
    main()
