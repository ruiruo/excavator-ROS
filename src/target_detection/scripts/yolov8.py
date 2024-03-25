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
import threading
import apriltag

import rospy
from geometry2.tf2_ros.src import tf2_ros
from geometry_msgs.msg import TransformStamped
from target_detection.msg import Box_Point, Target_Points, Key_Point, Coordinate_point, Coordinate_points


class TargetDetection:
    def __init__(self):

        # 加载参数
        weight_path = rospy.get_param('~weight_path', 
                                      "$(find target_detection)/weights/yolov8n-pose.onnx")
        pub_topic = rospy.get_param('~pub_topic', '/coordinate_transform/CoordinatePoint')
        self.half = rospy.get_param('~half', 'False')
        self.show = rospy.get_param('~show', 'False')# 显示实时画面
        self.conf = rospy.get_param('~conf', '0.5')
        self.point_search_range = rospy.get_param('~point_search_range', '4')
        self.model = YOLO(weight_path, task='pose')
        self.debug = rospy.get_param('~debug', 'False')

        # 选择yolo运行设备
        if (rospy.get_param('~cpu', 'False')):
            self.device = 'cpu'
        else:
            self.device = '0'

        self.TargetPoints = Target_Points()
        self.cam_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.color_cam_to_body_tf = TransformStamped()
        self.yaw = 0.
        self.seq = 0
        self.start_time = rospy.get_rostime()
        # 彩色相机内参按照图片旋转90度变换
        self.K = np.array([[622.461673, 0., 225.527207], 
                           [  0., 620.817214, 418.324731],[  0., 0., 1., ]], dtype=np.float64)
        self.inv_K = np.linalg.inv(self.K) 
        self.norm_y = np.array([[0.], [-1.], [0.]])
        self.calibrate_array = np.empty((3,0), dtype=np.float64)
        self.calibrate_array_size = 1.

        #相机图像pipeline配置
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        while True:
            try:
                    self.pipeline.start(config)#等待直至相机连接
                    break
            except Exception as e:
                rospy.logwarn_throttle_identical(60,"Please connect camera.")
        align_to = rs.stream.color
        align = rs.align(align_to)

        # 小臂apriltag配置
        self.forearm_detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))

        # tf坐标变换话题
        self.conversion_buffer = tf2_ros.Buffer(rospy.Time())
        self.conversion_listener = tf2_ros.TransformListener(self.conversion_buffer)

        #发布检测框与关键点话题
        self.position_pub = rospy.Publisher(pub_topic, Coordinate_points, queue_size=10)

        # 子线程用于接受加速度计数据
        accel_receive = threading.Event()
        accel_receive.set()
        accel_t = threading.Thread(target=self.accel_thread, args=(accel_receive,))
        accel_t.daemon = True
        accel_t.start()
        #子线程获取小臂apriltag
        forearm_receive = threading.Event()
        forearm_t = threading.Thread(target=self.forearm_thread, args=(forearm_receive,))
        forearm_t.daemon = True
        forearm_t.start()

        # 主循环
        while not rospy.is_shutdown() :
            # try:
            # 图像获取与预处理12ms 主要为旋转耗时8ms
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if  aligned_depth_frame and color_frame:
                    rospy.loginfo_throttle_identical(600,"Get image!")
                    self.TargetPoints = Target_Points()
                    self.coordinate_points = Coordinate_points()
                    self.coordinate_points.header.stamp = rospy.Time.now()
                    self.seq =+ 1
                    self.coordinate_points.header.seq = self.seq
                    self.coordinate_points.header.frame_id = "camera_color_frame"

                    self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    self.color_image = np.asanyarray(color_frame.get_data())
                    self.color_image = cv2.rotate(self.color_image, cv2.ROTATE_90_CLOCKWISE)
                    self.depth_image = cv2.rotate(self.depth_image, cv2.ROTATE_90_CLOCKWISE)
                    forearm_receive.set()
                    # YOLO推理与数据处理
                    self.predict()

                # 将校正后得到的相机外参用于关键点转换
                if((self.calibrate_array_size >= 250) and
                    (not self.conversion_buffer.can_transform("camera_color_frame", "body", rospy.Time()))):
                    self._calibrate()
                    #校准结束重置
                    accel_receive.clear()
                    self.calibrate_array = np.empty((3,0), dtype=np.float64)
                    self.calibrate_array_size = 1.

                # 使用相机外参对关键点转换并发布 耗时2-4ms
                if self.conversion_buffer.can_transform("camera_color_frame", "body",  rospy.Time()):
                    self._transform_kp()
                    self.position_pub.publish(self.coordinate_points)
            # except Exception as e:
            #     rospy.logwarn_throttle_identical(60,\
            #                                      "Unable to obtain camera image data, check if camera is working.")
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def predict(self):
            #输出模型监测结果，并导出预览图片与整理检测结果 使用engine half耗时39-44ms
            results = self.model(self.color_image, conf=self.conf, device=self.device, half=self.half)

            if self.show:
                image = results[0].plot()
                try:
                    if self.show:
                        cv2.imshow("yolov8 inf", image)
                        cv2.waitKey(1)
                except Exception as e:
                    pass

            # 数据整理与处理9-12ms 整理耗时约7ms
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

    def forearm_thread(self, forearm_receive):
        while not rospy.is_shutdown():
            while forearm_receive.is_set():
                gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
                tags = self.forearm_detector.detect(gray)
                for tag in tags:
                    if tag.tag_family == b'tag36h11' and tag.tag_id == 5:
                        angle_degrees = np.empty((1,0))
                        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(tag.homography, self.K)
                        for ns in Ns:
                            ns = np.array([0, ns[1][0], ns[2][0]])
                            ns_norm = np.linalg.norm(ns)
                            angle_radians = np.degrees(np.arccos(np.dot(ns, self.norm_y) / ns_norm))
                            angle_degrees = np.append(angle_degrees, angle_radians)
                        self.yaw = -np.sort(angle_degrees)[1]
                forearm_receive.clear()

    def accel_thread(self, accel_receive):
        while not rospy.is_shutdown():
        # 接收到加速度信息更新加速度数据
            while accel_receive.is_set():
                try:
                    frames = self.pipeline.wait_for_frames()
                    accel_frame = frames.first_or_default(rs.stream.accel)
                    if accel_frame:
                        accel_data = accel_frame.as_motion_frame().get_motion_data()
                        xyz = np.array([[accel_data.x], [accel_data.y], [accel_data.z]], dtype=np.float64)
                        if self.calibrate_array.size > 0:
                            # 使用递归算法更新calibrate_array
                            self.calibrate_array = self.calibrate_array +  \
                                (xyz - self.calibrate_array) / self.calibrate_array_size
                            self.calibrate_array_size += 1
                        else:
                            self.calibrate_array = xyz
                            self.calibrate_array_size = 1.
                except Exception as e:
                    rospy.logwarn_throttle_identical(60,\
                                                    "Unable to obtain camera aceel data, check if camera is working.")
                    break

    # 将关键点坐标通过(彩色相机->机体)旋转矩阵进行变换
    def _transform_kp(self): 
        for p in self.TargetPoints.key_points:
            coordinate_point = Coordinate_point()
            coordinate_point.name = p.name
            point = np.array([[p.x], [p.y], [1]])
            if p.distance != 0:
                point= np.dot(p.distance * self.inv_K , point)
                point[[1, 2]] = point[[2, 1]]
                coordinate_point.x, coordinate_point.y, coordinate_point.z = np.dot(
                    point.T,self.R_color_to_body.as_matrix()).T[[1, 2, 0]]
                coordinate_point.y = -coordinate_point.y
            #x,y,z 前 右 下
            self.coordinate_points.coordinate_points.append( coordinate_point)

    # 根据是否校准处理关键点数据
    def _data_process(self, boxes):
        for box in boxes:
            self._collect_boxes(box[:6])
            self._collect_keypoints(box[6:])

    # 处理物体框数据
    def _collect_boxes(self, box):
        BoxPoint = Box_Point(
            probability=box[1],
            distance=self._get_point_distance([int(box[2]), int(box[3])]),
            x=box[2], y=box[3],
            width=box[4],  height=box[5],
            name=box[0]
        )
        self.TargetPoints.box_points.append(BoxPoint)

    # 处理关键点数据
    def _collect_keypoints(self, keypoints):
        kp_name = ['A', 'B', 'A_d', 'B_d', 'BF_l', 'BF_r']
        self.TargetPoints.key_points.extend(
            Key_Point(
                x=kp[0], y=kp[1],
                distance=self._get_point_distance([int(kp[0]), int(kp[1])]),
                name=name
            ) for name, kp in zip(kp_name, keypoints) if self._get_point_distance([int(kp[0]), int(kp[1])]) != 0.
        )

    # 获取指定关键点距离，距离单位cm 原始数据单位为mm
    def _get_point_distance(self, mid_pos):
        distance_list = np.empty((1, 0))
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
        return np.round(np.mean(distance_list) / 10, 4)
        
    # RGBD相机外参旋转矩阵获取
    def _calibrate(self):
        calibrate_angle = self.calibrate_array.T
        calibrate_angle = np.clip(calibrate_angle, -C.g, C.g) 
        roll = math.degrees(math.atan(-calibrate_angle[0][0] /         # roll正方向后变前
                                        math.sqrt(calibrate_angle[0][1]**2 + calibrate_angle[0][2]**2)
                                        if calibrate_angle[0][1]**2 + calibrate_angle[0][2]**2 != 0. else float('inf')))
        # pitch正方向右变左 仅保证相机平放时相机向下90度内的正确性
        pitch =  90. + math.degrees(math.atan(calibrate_angle[0][1] /      
                                                math.sqrt(calibrate_angle[0][0]**2 + calibrate_angle[0][2]**2) 
                                                if calibrate_angle[0][0]**2 + calibrate_angle[0][2]**2 != 0. else float('inf')))
        yaw = self.yaw
        rospy.loginfo("Calibration successful! ")
        rospy.loginfo("pitch = %2f  roll = %2f yaw = %2f", pitch, roll, yaw)
        self._color_to_body_coordinate_system(pitch, roll, yaw)

    # (彩色相机->机体)旋转矩阵计算与发布 传入的pitch, roll, yaw分别为左前下
    def _color_to_body_coordinate_system(self, pitch, roll, yaw):
        R_body_to_color_yaw = R.from_euler('XYZ', [0, 0, yaw], degrees=True)
        # 加速度计坐标到相机坐标系仅有平移关系
        R_aceel_to_color = R.from_quat([0.00121952651534, -0.00375633803196, 
                                         -0.000925257743802, 0.999991774559])
        R_world_to_accel = R.from_euler('XYZ', [pitch, roll, 0], degrees=True)
        R_world_to_color = R_world_to_accel * R_aceel_to_color
        #缺少body与world的旋转矩阵 yaw直接参考机体 roll与pitch参考世界坐标系
        self.R_color_to_body = ((R_world_to_color * R_body_to_color_yaw).inv())

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

    # 测试指定函数段耗时辅助函数
    def _time_test(self, key=1):
        if key == 1:
            self.start_time = rospy.get_rostime()
        if key == 2:
            time1 = (rospy.get_rostime().secs - self.start_time.secs)*1000 + (rospy.get_rostime().nsecs - self.start_time.nsecs)/1000000
            rospy.loginfo(time1, "ms")

 
def main():
    rospy.init_node('target_detection', anonymous=True)
    yolo_dect = TargetDetection()
    rospy.spin()


if __name__ == "__main__":
    main()
