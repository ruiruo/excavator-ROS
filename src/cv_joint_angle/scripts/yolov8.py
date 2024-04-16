#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as R
import scipy.constants as C

import multiprocessing
import ast, glob, os, time

import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import apriltag

import rospy
from geometry2.tf2_ros.src import tf2_ros
from geometry_msgs.msg import TransformStamped
from cv_joint_angle.msg import Joint_angle


class TargetDetection:
    def __init__(self):

        # 加载YOLO参数
        weight_path = rospy.get_param('~weight_path', 
                                      "$(find cv_joint_angle)/weights/yolov8n-pose.onnx")
        self.half = rospy.get_param('~half', 'False')
        self.show = rospy.get_param('~show', 'False')
        self.conf = rospy.get_param('~conf', '0.5')
        self.model = YOLO(weight_path, task='pose')
        self.classes = (None if rospy.get_param('~yolo_classes', 'None') == 'None' 
                        else ast.literal_eval(rospy.get_param('~yolo_classes', 'None')))
        self.target_box = rospy.get_param('~target_box', 'False')
        self.device = 'cpu' if rospy.get_param('~cpu', 'False') == True else '0'

        # 姿态获取与校正参数
        self.point_search_range = int(rospy.get_param('~point_search_range', '4'))
        self.center2A = np.array(ast.literal_eval(
            rospy.get_param('~center2A_offset', '[1.25, 1.1, 1.65]')), dtype=np.float64)
        self.B_error = float(rospy.get_param('~B_tolerance_scope', '1.15'))
        self.quadrilateral_length = np.array(ast.literal_eval(rospy.get_param(
            '~quadrilateral_side_length', '[3.2, 5.35, 5.25, 4.3]')), dtype=np.float64)
        self.A2Adown = self.quadrilateral_length[0]
        self.A2B = self.quadrilateral_length[1]
        self.accel_calibrate_array_size = 0
        self.accel_calibrate_array = np.array([0.,0.,0.])
        
        # 调试参数
        self.debug = rospy.get_param('~debug', 'False') == 'True'

        # ROS话题相关参数
        self.tran_flag = False
        self.cam_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.color_cam_to_body_tf = TransformStamped()
        self.start_time = rospy.get_time()
        self.seq = 0
        # tf坐标变换话题
        self.conversion_buffer = tf2_ros.Buffer(rospy.Time())
        self.conversion_listener = tf2_ros.TransformListener(self.conversion_buffer)
        #发布倾角话题
        pub_topic = rospy.get_param('~pub_topic', '/cv_joint_angle')
        self.angle_pub = rospy.Publisher(pub_topic, Joint_angle, queue_size=1)

        #相机参数
        self.K = np.array([[622.461673, 0., 225.527207], 
                           [  0., 620.817214, 418.324731],[  0., 0., 1.]], dtype=np.float64)
        self.inv_K = np.linalg.inv(self.K) 
        #相机图像pipeline配置
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
        while True:
            try:
                    pipeline.start(config)#等待直至相机连接
                    break
            except Exception as e:
                rospy.logwarn_throttle_identical(60,"Please connect camera.")
        rospy.loginfo_once("Successfully connected camera.")
        align_to = rs.stream.color
        align = rs.align(align_to)

        #多进程参数
        self.apriltag_bag = multiprocessing.Array('f', 8)
        self.yaw = multiprocessing.Value('f', 0)
        self.time_stamp = multiprocessing.Value('i', 0)

        # 小臂apriltag配置
        self.forearm_detector = apriltag.Detector(apriltag.DetectorOptions(
            families="tag36h11", refine_pose=True))#开启姿态解算优化后旋转矩阵默认旋转顺序XYZ
        #子进程获取小臂apriltag
        forearm_receive = multiprocessing.Event()
        yaw_calibrate = multiprocessing.Event()
        yaw_calibrate.clear()
        forearm_p = multiprocessing.Process(target=self.forearm_process, \
                                            args=(forearm_receive, self.yaw, yaw_calibrate, self.apriltag_bag, self.time_stamp))
        forearm_p.start()
        self._clear_npy()

        # 主循环
        while not rospy.is_shutdown() :
            self.tran_flag = self.conversion_buffer.can_transform("camera_color_frame", "body", rospy.Time())
            try:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                accel_frame = frames.first_or_default(rs.stream.accel)

                if  aligned_depth_frame and color_frame:
                    rospy.loginfo_once("Get image!")
                    #创建发布信息并赋值Header
                    self.joint_angle = Joint_angle()
                    self.joint_angle.header.stamp = rospy.Time.now()
                    self.seq += 1
                    self.joint_angle.header.seq = self.seq
                    self.joint_angle.header.frame_id = "camera_color_frame"

                    #预处理RGBD信息并保存nparray同步给子进程
                    time_stamp = self.joint_angle.header.stamp.nsecs
                    self.time_stamp.value = time_stamp
                    color_image = np.asanyarray(color_frame.get_data())
                    self.color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
                    color_image_name = f"{time_stamp}_color.npy"
                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    self.depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
                    depth_image_name = f"{time_stamp}_depth.npy"
                    np.save(color_image_name, self.color_image)
                    np.save(depth_image_name, self.depth_image)
                    if os.path.exists(color_image_name) and os.path.exists(depth_image_name):
                        forearm_receive.set()

                #收集加速度计信息
                if self.accel_calibrate_array_size < 250 and not self.tran_flag:
                    self._accecl_collect(accel_frame)
                # 将校正后得到的相机外参用于关键点转换
                if self.accel_calibrate_array_size >= 250 and yaw_calibrate.is_set() and not self.tran_flag:
                    self._calibrate()

                # 得到相机外参后计算各关节倾角并发布
                if self.tran_flag:
                    #与apriltag多进程同步TX2运行耗时45-50ms
                    yolo_result = self._transform_kp(self._predict())
                    result = self._bucket_attitude(yolo_result[:2,:])
                    if result is not None:
                        self.joint_angle.forearm, self.joint_angle.bucket = result
                    else:
                         rospy.logwarn_throttle_identical(1, "No data available")
                    self.angle_pub.publish(self.joint_angle)
                    rospy.loginfo_throttle_identical(60, "Node is running...")
                
            except Exception as e:
                rospy.logwarn_throttle_identical(60,\
                                                 "Unable to obtain camera image data, check if camera is working.")
                
        #结束子进程, 清空文件窗口, 停止接受相机信息
        forearm_p.terminate()
        self._clear_npy()
        cv2.destroyAllWindows()
        pipeline.stop()

    def forearm_process(self, forearm_receive, yaw, yaw_cal, apriltag_bag, time_stamp):
        '''
        这个进程负责检测apriltag并计算姿态
        校正完成前: 持续更新相机与机体的yaw差值
        校正完成后: 持续更新小臂方位角与A、A_down关键点三维信息

        @param forearm_receive: 被设置时进程将持续更新数据

        @param yaw: 机体与相机yaw夹角估计值
        @type  name: multiprocessing.Value

        @param yaw_cal: 被设置时表示yaw校准合格

        @param apriltag_bag: 最新小臂姿态信息
        @type  name: multiprocessing.Array

        @param time_stamp: 主进程RGBD数据时间戳
        @type  name: multiprocessing.Value
        '''
        # 子进程运行TX2耗时22-25ms
        yaw_count = 1.
        while not rospy.is_shutdown():
            while forearm_receive.is_set():
                images = self._get_image_files(time_stamp.value)
                if images is None:
                    break
                color_image, depth_image = images 
                self._clear_npy()

                # yuv_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2YUV)
                # yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
                # gray = yuv_image[:, :, 0]
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                forearm_list = self._apriltag_predict(gray, yaw, yaw_cal, yaw_count)
                
                if yaw_cal.is_set() and forearm_list is not None:
                    forearm_info = self._get_forearm_info(forearm_list, depth_image)
                    if forearm_info is None:
                        break
                    apriltag_bag[0] = time_stamp.value
                    apriltag_bag[1] = forearm_info[0]
                    apriltag_bag[2] = forearm_info[1][0]
                    apriltag_bag[3] = forearm_info[1][1]
                    apriltag_bag[4] = forearm_info[1][2]
                    apriltag_bag[5] = forearm_info[2][0]
                    apriltag_bag[6] = forearm_info[2][1]
                    apriltag_bag[7] = forearm_info[2][2]
                forearm_receive.clear()

    def _get_forearm_info(self, forearm_list, depth_image):
        '''
        通过forearm_list中小臂关键点的坐标值输出小臂姿态与A, A_down关键点坐标

        @param forearm_list: 指定关键点的像素坐标
        @type forearm_list: np.array

        @param depth_image: 深度图像数据
        @type depth_image: np.array

        @return: 小臂的滚动角度以及A和A_down关键点的坐标。
        @rtype: tuple

        该函数首先计算深度图像中指定像素坐标处的距离, 然后转换为实际坐标
        接着, 它计算小臂的滚动角度, 并使用这个角度来确定A和A_down关键点的位置
        如果在计算过程中发现Apriltag标记错误, 则函数将返回None
        '''
        forearm_list = self._get_distance(forearm_list, depth_image)
        forearm_coordinate = self._transform_kp(forearm_list)
        center = forearm_coordinate[0, 0] if forearm_coordinate[0, 0, 2] > forearm_coordinate[0, 1, 2] else forearm_coordinate[0, 1]
        if (forearm_coordinate[0, 0, :] == forearm_coordinate[0, 1, :]).all():
            forearm_coordinate = forearm_coordinate[1:, :, :]
        vector_diff = forearm_coordinate[:, 0, :] - forearm_coordinate[:, 1, :]
        if np.any(vector_diff[:, 0] == 0):
            rospy.logwarn_throttle_identical(1, "Apriltag is wrong")
            return None
        forearm_roll_list = np.degrees(np.arctan(vector_diff[:, 2] / vector_diff[:, 0]))
        forearm_roll = np.mean(np.sort(forearm_roll_list))
        roll_sin = np.sin(np.deg2rad(forearm_roll))
        roll_cos = np.cos(np.deg2rad(forearm_roll))
        A_apriltag = center + np.dot(
            self.center2A, np.array([[roll_sin, 0, roll_cos], [0, 1, 0], [roll_cos, 0, -roll_sin]], dtype=np.float64))
        Adown_apriltag = A_apriltag + np.array(
            [roll_cos * self.A2Adown, 0, roll_sin * self.A2Adown], dtype=np.float64)
        return forearm_roll, A_apriltag, Adown_apriltag

    def _apriltag_predict(self, image, yaw, yaw_cal, yaw_count):
        '''
        通过检测图像中的Apriltag标记来预测小臂姿态, 并更新yaw值

        @param image: 输入图像, 用于检测Apriltag标记
        @type image: np.array

        @param yaw: 当前的yaw值, 将根据检测到的Apriltag标记进行更新
        @type yaw: multiprocessing.Value

        @param yaw_cal: 一个事件标志, 用于指示是否已经校准了yaw值
        @type yaw_cal: threading.Event

        @param yaw_count: 用于平滑yaw值更新的计数器
        @type yaw_count: float

        @return: 如果检测到Apriltag标记且未校准yaw值, 则返回None; 否则返回更新的forearm_list
        @rtype: np.array 或 None

        该函数首先尝试在输入图像中检测Apriltag标记 
        如果在一定时间内未检测到标记, 将发出警告
        如果检测到标记, 将根据标记的homography矩阵计算小臂的滚动角度, 并更新yaw值
        如果yaw值在特定范围内并且与第一个角度的差异小于1度, 则设置yaw_cal标志
        '''
        forearm_list = None
        tags = self.forearm_detector.detect(image)
        if tags == [] and rospy.get_time()-self.start_time>5:
                rospy.logwarn_throttle_identical(5, "No apriltag detected")
                return None
        else:
            num = len(tags)
            rospy.logdebug_throttle(10, "Have %d apriltag(s) detected", num)
        for tag in tags:
            if tag.tag_family == b'tag36h11' and tag.tag_id == 5:
                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(tag.homography, self.K)
                if not yaw_cal.is_set():
                    angle = [-np.degrees(np.arctan(r[1, 0] / r[0, 0])) for r in Rs[::2]]
                    first_angle = float(angle[0]) if angle else None
                    if   first_angle is None:
                        break
                    if math.isclose(yaw_count, 1.):
                        yaw.value = first_angle
                    else:
                        yaw.value += (first_angle - yaw.value) / yaw_count
                        if yaw_count>100:
                            yaw_count = 1.
                    yaw_count += 1
                    if 15 < yaw.value < 30 and abs(yaw.value - first_angle) < 1:
                        yaw_cal.set()
                        yaw_count = 1.
                        return None
                else:
                    if forearm_list is not None:
                        forearm_list[0, 0, :] = tag.center
                        forearm_list = np.vstack([forearm_list, 
                                                np.expand_dims(np.array([tag.corners[0], tag.corners[3]]), axis=0), 
                                                np.expand_dims(np.array([tag.corners[1], tag.corners[2]]), axis=0)])
                    else:
                        forearm_list = np.array([np.array([tag.center, tag.center]),
                                                np.array([tag.corners[0], tag.corners[3]]), 
                                                np.array([tag.corners[1], tag.corners[2]])])
        return forearm_list

    def _get_image_files(self, time_stamp):
        '''
        根据时间戳获取对应的彩色图像和深度图像文件

        @param time_stamp: 需要获取图像的时间戳
        @type time_stamp: int

        @return: 如果找到对应时间戳的图像, 则返回彩色图像和深度图像; 否则返回None
        @rtype: tuple 或 None

        该函数首先搜索所有的.npy文件, 然后根据文件名中的时间戳将它们分组
        如果找到与给定时间戳匹配的图像文件, 它将加载这些文件并返回
        如果在一定时间内未找到匹配的文件, 将发出警告
        '''
        color_image = None
        depth_image = None
        npy_files = glob.glob('*.npy')
        if len(npy_files)>=2:
            split_names = [os.path.splitext(f)[0].split('_') for f in npy_files]
            grouped_files = {}
            for name_parts in split_names:
                if name_parts[0] not in grouped_files:
                    grouped_files[name_parts[0]] = []
                grouped_files[name_parts[0]].append('_'.join(name_parts))
            for timestamp, files in grouped_files.items():
                if int(timestamp) !=  time_stamp:
                    continue
                for file in files:
                    if 'color' in file:
                        color_image = np.load(f"{file}.npy")
                    elif 'depth' in file:
                        depth_image = np.load(f"{file}.npy").astype(np.float64)
        if rospy.get_time()-self.start_time>5:
            if len(npy_files)<2 :
                rospy.logwarn_throttle_identical(5, "Forearm_process didn't find picture")
                return None
            elif color_image is None or depth_image is None:
                rospy.logwarn_throttle_identical(5, 
                                                "Forearm_process cannot find an image with a suitable timestamp")
                self._clear_npy()
                return None
        else :
            if len(npy_files)<2 or color_image is None or depth_image is None:
                self._clear_npy()
                return None
        return color_image, depth_image

    def _accecl_collect(self, accel_frame):
        '''
        收集加速度数据并更新校准数组。

        此方法接收一个加速度帧，从中提取加速度数据，并更新校准数组。
        如果校准数组尚未初始化（即大小为零），则直接将当前帧的加速度数据作为初始值。
        如果校准数组已经有值，则使用递归平均算法更新数组，以便平滑地集成新的加速度数据。

        @param accel_frame: 加速度帧，包含当前时刻的加速度传感器数据。
        '''
        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            xyz = np.array([accel_data.x, accel_data.y, accel_data.z])
            if self.accel_calibrate_array_size > 0:
                # 使用递归算法更新calibrate_array
                self.accel_calibrate_array = self.accel_calibrate_array + \
                    (xyz - self.accel_calibrate_array) / self.accel_calibrate_array_size
                self.accel_calibrate_array_size += 1
            else:
                self.accel_calibrate_array = xyz
                self.accel_calibrate_array_size = 1

    def _predict(self):
            '''
            将当前图像送入YOLO推理并进行输出结果的数据整理

            @return: 推理获取的关键点像素坐标
            '''
            #输出模型监测结果，并导出预览图片与整理检测结果 使用engine half耗时39-44ms
            results = self.model(self.color_image, conf=self.conf, device=self.device, half=self.half, classes=self.classes)
            if self.show:
                image = results[0].plot()
                try:
                    if self.show:
                        cv2.imshow("yolov8 inf", image)
                        cv2.waitKey(1)
                except Exception as e:
                    pass

            if any(results[0].boxes.cls):
                num = len(results[0].boxes.cls)
                if self.target_box:
                    boxes = torch.stack([torch.cat((cls, conf, xywh), dim=0)\
                                    for cls, conf, xywh in zip(
                                        results[0].boxes.cls.reshape(num, 1),
                                        results[0].boxes.conf.reshape(num, 1),
                                        results[0].boxes.xywh)])
                if len(results[0].boxes.cls) > 1:
                    keypoints = torch.stack([*results[0].keypoints.xy], dim=1).view(-1, num, 2)
                else:
                    keypoints = (results[0].keypoints.xy).unsqueeze(0).view(-1, num, 2)
                return self._get_distance(keypoints.numpy())
            return None

    def _transform_kp(self, kp): 
        '''
        通过(彩色相机->机体)旋转矩阵变换关键点坐标.

        @param kp: 包含关键点三维坐标的, 形状为 (n, m, 3).
                        'n' 代表不同种类的数量, 'm' 代表同一种类中关键点的数量.
        @type kp: np.array

        @return: 变换后的关键点坐标, 形状与输入相同.
        @rtype: np.array 或 None

        如果输入的 kp 不为空, 此函数将对每个关键点应用旋转变换, 
        并返回变换后的坐标. 如果输入为空, 则返回 None. 
        '''
        if kp is not None:
            transform_kp = np.zeros((kp.shape[0], kp.shape[1], 3), dtype=np.float64)
            for row in range(kp.shape[0]):
                for col in range(kp.shape[1]):
                    point = np.array([[kp[row, col, 0]], [kp[row, col, 1]], [1.]])
                    point = kp[row, col, 2] * np.dot(self.inv_K, point)
                    if self.tran_flag:
                        point = np.dot(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64), point)
                        transform_kp[row, col] = np.dot(point.T, self.R_color_to_body).ravel()
                    else:
                        transform_kp[row, col] = np.dot(
                            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64), point).ravel()
            # x,y,z 前 右 下
            return transform_kp
        return None

    def _bucket_attitude(self, AB_kps):
        '''
        计算铲斗的方位角, 并结合小臂的方位角输出

        @param AB_kps:包含A、B关键点的(2, m, 3), 其中m为相同种类点的个数
        @type AB_kps:np.array
        
        @return: 返回一个包含小臂方位角和铲斗方位角的元组
        @rtype: tuple(np.array) 或 None

        此函数首先检查是否有有效的 AprilTag 数据包
        如果没有, 将记录警告并返回 None
        如果有新的 AprilTag 更新, 将计算并返回小臂和铲斗的方位角
        '''
        apriltag_msg = [*self.apriltag_bag]
        if abs(self.joint_angle.header.stamp.nsecs - apriltag_msg[0])>1000:
            rospy.logwarn_throttle_identical(5, "No new aprilatg updates.")
            return None
        point_A = apriltag_msg[2:5]
        point_A_down = apriltag_msg[5:]
        point_B = self._correction_point_B(point_A, AB_kps)
        point_B_down = self._calculate_opposite_quadrilateral_vertex(point_A_down, point_B)
        A2B_down_distance = np.linalg.norm(point_A - point_B_down)
        angle_NQK = np.degrees(np.arccos((
            self.quadrilateral_length[0]**2 + self.quadrilateral_length[3]**2 - A2B_down_distance)/
            (2 * self.quadrilateral_length[0] * self.quadrilateral_length[3])))
        bucket_roll = float(angle_NQK - apriltag_msg[1] + 5.5 + 108)
        return apriltag_msg[1], bucket_roll
        
    def _calculate_opposite_quadrilateral_vertex(self, vertex_A, vertex_B):
        '''
        计算由两个圆心和半径定义的两个圆的交点，这两个交点代表四边形的对角顶点

        @param vertex_A: 第一个圆的圆心坐标
        @param vertex_B: 第二个圆的圆心坐标
        @type vertex_A, vertex_B: np.array

        @return: 位于更高z坐标的交点, 代表四边形的对角顶点
        @rtype: np.array

        此函数通过计算两个圆的交点来确定四边形的一个未知顶点
        交点是通过解析几何方法得出的
        '''
        radius_A = self.quadrilateral_length[3]
        radius_B = self.quadrilateral_length[2]
        center_distance = np.linalg.norm(vertex_B - vertex_A)
        a = (radius_A**2 - radius_B**2 + center_distance**2) / (2 * center_distance)
        h = np.sqrt(radius_A**2 - a**2)
        midpoint = vertex_A + a * (vertex_B - vertex_A) / center_distance
        intersection1 = midpoint + h * np.array(
            [-1 * (vertex_B[2] - vertex_A[2]) / center_distance, 0, (vertex_B[0] - vertex_A[0]) / center_distance], dtype=np.float64)
        intersection2 = midpoint - h * np.array(
            [-1 * (vertex_B[2] - vertex_A[2]) / center_distance, 0, (vertex_B[0] - vertex_A[0]) / center_distance], dtype=np.float64)
        return intersection1 if intersection1[2] > intersection2[2] else intersection2

    def _correction_point_B(self, A_apriltag, AB_yolo_kps):
        '''
        通过apriltag_A与yolo推理A偏差来纠正yolo推理B, 并将B与apriltag_A对齐同一y平面

        @param A_apriltag: apriltag检测得到的A点的三维坐标。
        @param AB_yolo_kps: YOLO检测得到的A、B两点的三维坐标, 形状为 (2, m, 3), 其中m为同类点的数量
        @type A_apriltag, AB_yolo_kps: np.array

        @return: 校正后的B点三维信心
        @rtype: np.array

        该函数首先计算A点的y坐标与YOLO关键点y坐标的比例, 以此来调整AB_yolo_kps,
        使得B点的y坐标与A点对齐. 然后, 根据A点和B点的z坐标差异, 进一步调整B点的z坐标,
        以确保B点的准确性
        '''
        ac = (A_apriltag[1] / AB_yolo_kps[:, :, 1, np.newaxis]) * AB_yolo_kps - A_apriltag
        ac_normalized = ac / np.linalg.norm(ac, axis=2, keepdims=True)
        AB_yolo_kps = A_apriltag + self.A2B * ac_normalized
        A_yolo, B_yolo = np.mean(AB_yolo_kps, axis=1)
        A_delta_z = np.clip(A_apriltag[2] - A_yolo[2], -self.B_error/2., self.B_error/2.)
        B_yolo[2] += A_delta_z
        ab_vector = B_yolo - A_apriltag
        ab_vector_normalized = ab_vector / np.linalg.norm(ab_vector)
        B_yolo_new = A_apriltag + self.A2B * ab_vector_normalized
        return B_yolo_new
            
    def _get_distance(self, kp, depth_img=None):
        '''
        获取关键点距离数值并拼接进输入

        @param kp: 包含关键点三维坐标, 形状为 (n, m, 2).
                        'n' 代表不同种类的数量, 'm' 代表同一种类中关键点的数量.
        @type kp: np.array

        @return: 获得距离的关键点坐标, 形状为 (n, m, 3).
        @rtype: np.array
        '''
        if depth_img is None:
            depth_img = self.depth_image
        add_distance_array = np.zeros((kp.shape[0], kp.shape[1], 3), dtype=np.float64)
        for row in range(kp.shape[0]):
            for col in range(kp.shape[1]):
                mid_pos = kp[row, col]
                distance = self._get_point_distance(mid_pos.astype(int), depth_img)
                add_distance_array[row, col] = np.concatenate((mid_pos, distance))
        return add_distance_array

    def _get_point_distance(self, pixel_coordinates, depth_image):
        '''
        计算指定像素坐标处关键点的距离, 单位为厘米, 原始深度图像数据单位为毫米

        @param pixel_coordinates: 指定关键点的像素坐标
        @type pixel_coordinates: np.array

        @param depth_image: 深度图像数据
        @type pixel_coordinates: np.array

        @return: 计算得到的关键点距离
        @rtype: np.array

        此函数在深度图像中搜索指定像素坐标周围的非零距离值,
        并计算这些值的统计量(如四分位数)来估算关键点的距离
        如果搜索范围内没有非零值, 则返回默认距离
        '''
        distance_list = np.array([])
        for scope in range(self.point_search_range):
            if (pixel_coordinates[0] + scope < depth_image.shape[1] and 
                pixel_coordinates[1] + scope < depth_image.shape[0]):
                distance_list = depth_image[
                    pixel_coordinates[1]-scope:pixel_coordinates[1]+scope+1,
                    pixel_coordinates[0]-scope:pixel_coordinates[0]+scope+1]
                if np.sum(distance_list) != 0:
                    break
        distance_list = distance_list[distance_list != 0]
        if len(distance_list) >= 4:
            distance_list = np.quantile(distance_list, [0.25, 0.5, 0.75])
        elif len(distance_list) == 0:
            return np.array([10.])
        return np.array([np.round(np.mean(distance_list) * 1000) / 10000])
        
    def _calibrate(self):
        '''
        获取RGBD相机的外参旋转矩阵。

        此函数计算从加速度计坐标系到相机坐标系的旋转矩阵，
        并根据加速度计数据校准相机的姿态。
        '''
        # 从四元数转换为欧拉角，表示加速度计到彩色相机坐标系的旋转
        accel_to_color_rotation = R.from_quat([
            0.00121952651534, -0.00375633803196, 
            -0.000925257743802, 0.999991774559
        ]).as_euler('XYZ', degrees=True)

        calibrated_accel = np.clip(self.accel_calibrate_array, -C.g, C.g) 
        # 计算当前姿态与正下方向的差值，右侧为正方向
        roll_angle = -90 - math.degrees(math.atan(
            calibrated_accel[0] / 
            (math.sqrt(calibrated_accel[1]**2 + calibrated_accel[2]**2) or float('inf'))))
        # 计算当前姿态与正前方向的差值，前侧为正方向
        pitch_angle = math.degrees(math.atan(
            calibrated_accel[1] / 
            (math.sqrt(calibrated_accel[0]**2 + calibrated_accel[2]**2) or float('inf'))))

        self.roll = roll_angle + accel_to_color_rotation[1]
        pitch = pitch_angle + accel_to_color_rotation[0]
        yaw = self.yaw.value

        # 计算并发布(彩色相机->机体)旋转矩阵
        color_to_body_rotation = R.from_euler('XYZ', [pitch, self.roll, yaw], degrees=True)
        self.R_color_to_body = np.array(color_to_body_rotation.as_matrix(), dtype=np.float64)
        self.color_cam_to_body_tf.header.frame_id = "camera_color_frame"
        self.color_cam_to_body_tf.header.stamp = rospy.Time.now()
        self.color_cam_to_body_tf.child_frame_id = "body"
        self.color_cam_to_body_tf.transform.translation.x = 0.
        self.color_cam_to_body_tf.transform.translation.y = 0.
        self.color_cam_to_body_tf.transform.translation.z = 0.
        self.color_cam_to_body_tf.transform.rotation.x = color_to_body_rotation.as_quat()[0]
        self.color_cam_to_body_tf.transform.rotation.y = color_to_body_rotation.as_quat()[1]
        self.color_cam_to_body_tf.transform.rotation.z = color_to_body_rotation.as_quat()[2]
        self.color_cam_to_body_tf.transform.rotation.w = color_to_body_rotation.as_quat()[3]
        self.cam_broadcaster.sendTransform(self.color_cam_to_body_tf)

        rospy.loginfo(f"Calibration successful! \n pitch: {pitch}, roll: {self.roll}, yaw: {yaw}")

    def _clear_npy(self):
        '''
        清除进程间通信的图片nparray文件
        '''
        npy_files = glob.glob('*.npy')
        for f in npy_files:
            os.remove(f)

    def _time_test(self, key=1):
        '''
        测试指定函数段耗时辅助函数, 并通过info等级消息输出结果

        @param key:1为开始计时, 2为结束计时
        @type: int
        '''
        if key == 1:
            self.start_time = time.time()
        if key == 2:
            end_time = time.time()
            time_elapsed = (end_time - self.start_time) * 1000 
            rospy.loginfo(f"{time_elapsed} ms")

 
def main():
    debug = rospy.get_param('~debug', 'False') == 'True'
    if debug:
        rospy.init_node('cv_joint_angle', log_level=rospy.DEBUG, anonymous=True)
    else:
        rospy.init_node('cv_joint_angle', log_level=rospy.INFO, anonymous=True)
    cv_joint_angle = TargetDetection()
    rospy.spin()


if __name__ == "__main__":
    main()
