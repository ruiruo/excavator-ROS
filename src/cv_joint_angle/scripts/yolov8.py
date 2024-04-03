#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as R
import scipy.constants as C

import multiprocessing, threading
import ast

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
        print(rospy.get_param('~cpu', 'False'))
        self.device = 'cpu' if rospy.get_param('~cpu', 'False') == True else '0'

        # 姿态获取与校正参数
        self.point_search_range = int(rospy.get_param('~point_search_range', '4'))
        self.center2A = torch.tensor(ast.literal_eval(
            rospy.get_param('~center2A_offset', '[1.25, 1.1, 1.65]')), dtype=torch.float64)
        self.B_error = float(rospy.get_param('~B_tolerance_scope', '1.15'))
        self.quadrilateral_length = torch.tensor(ast.literal_eval(rospy.get_param(
            '~quadrilateral_side_length', '[3.2, 5.35, 5.25, 4.3]')), dtype=torch.float64)
        self.A2Adown = self.quadrilateral_length[0]
        self.A2B = self.quadrilateral_length[1]
        
        # 发布话题名称
        pub_topic = rospy.get_param('~pub_topic', '/cv_joint_angle')
        
        # 调试参数
        self.debug = rospy.get_param('~debug', 'False') == 'True'
        if self.debug:
            torch.set_printoptions(sci_mode=False)

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
        self.angle_pub = rospy.Publisher(pub_topic, Joint_angle, queue_size=1)

        #相机参数
        self.K = np.array([[622.461673, 0., 225.527207], 
                           [  0., 620.817214, 418.324731],[  0., 0., 1., ]], dtype=np.float64)
        self.K_tensor = torch.from_numpy(self.K)
        self.inv_K = np.linalg.inv(self.K) 
        #相机图像pipeline配置
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
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
        self.accel_calibrate_array = multiprocessing.Array('f', 3)
        self.accel_calibrate_array_size = multiprocessing.Value('i', 0)
        # 子进程用于接受加速度计数据
        accel_receive = multiprocessing.Event()
        accel_receive.set()
        accel_p = multiprocessing.Process(target=self.accel_process, \
                                          args=(accel_receive, self.accel_calibrate_array, self.accel_calibrate_array_size,))
        accel_p.start()

        # 多线程参数
        self.apriltag_bag = None
        self.forearm_thread_record_time = None
        self.yaw_calibrate = False
        # 小臂apriltag配置
        self.forearm_detector = apriltag.Detector(apriltag.DetectorOptions(
            families="tag36h11", refine_pose=True))#开启姿态解算优化后旋转矩阵默认旋转顺序XYZ
        #子线程获取小臂apriltag
        forearm_receive = threading.Event()
        forearm_t = threading.Thread(target=self.forearm_thread, args=(forearm_receive, ))
        forearm_t.daemon = True
        forearm_t.start()

        # 主循环
        while not rospy.is_shutdown() :
                self.tran_flag = self.conversion_buffer.can_transform("camera_color_frame", "body", rospy.Time())
            # try:
            # 图像获取与预处理12ms 主要为旋转耗时8ms
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if  aligned_depth_frame and color_frame:
                    rospy.loginfo_throttle_identical(600,"Get image!")
                    self.joint_angle = Joint_angle()
                    self.joint_angle.header.stamp = rospy.Time.now()
                    self.seq =+ 1
                    self.joint_angle.header.seq = self.seq
                    self.joint_angle.header.frame_id = "camera_color_frame"

                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    self.color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
                    depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
                    depth_image = depth_image.astype(np.float32)
                    self.depth_image = torch.from_numpy(depth_image)
                    forearm_receive.set()

                # 将校正后得到的相机外参用于关键点转换
                if self.accel_calibrate_array_size.value >= 250 and self.yaw_calibrate and not self.tran_flag:
                    accel_receive.clear()
                    self._calibrate()
                    self.accel_calibrate_array = multiprocessing.Array('f', 3)
                    self.accel_calibrate_array_size = multiprocessing.Value('i', 0)

                # 得到相机外参后计算各关节倾角并发布
                if self.tran_flag:
                    yolo_result = self._transform_kp(self._predict())
                    self.joint_angle.forearm, self.joint_angle.bucket = self._bucket_attitude(yolo_result[:2,:])
                    self.angle_pub.publish(self.joint_angle)
                    rospy.loginfo_throttle_identical(60, "Node is running...")
            # except Exception as e:
            #     rospy.logwarn_throttle_identical(60,\
            #                                      "Unable to obtain camera image data, check if camera is working.")
        cv2.destroyAllWindows()
        pipeline.stop()
        accel_p.terminate()()

    def forearm_thread(self, forearm_receive):
        '''
        这个线程负责检测apriltag并计算姿态
        校正完成前: 持续更新相机与机体的yaw差值
        校正完成后: 持续更新小臂方位角与A、A_down关键点三维信息

        @param forearm_receive: 被设置时线程将持续更新数据
        '''
        yaw_count = 1.
        while not rospy.is_shutdown():
            while forearm_receive.is_set():
                # yuv_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2YUV)
                # yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
                # gray = yuv_image[:, :, 0]
                gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
                tags = self.forearm_detector.detect(gray)
                if tags == [] and rospy.get_time()-self.start_time>5:
                        rospy.logwarn_throttle_identical(5, "No apriltag detected")
                        break
                else:
                    num = len(tags)
                    rospy.logdebug_throttle(10, "Have %d apriltag(s) detected", num)
                
                forearm_list = None
                for tag in tags:
                    if tag.tag_family == b'tag36h11' and tag.tag_id == 5:
                        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(tag.homography, self.K)
                        if not self.tran_flag:
                            angle = [-np.degrees(np.arctan(r[1, 0] / r[0, 0])) for r in Rs[::2]]
                            first_angle = angle[0] if angle else 0  
                            if math.isclose(yaw_count, 1.):
                                self.yaw = first_angle
                            else:
                                self.yaw += (first_angle - self.yaw) / yaw_count
                                if yaw_count>100:
                                    yaw_count = 1.
                            yaw_count += 1
                            if 15 < self.yaw < 30 and abs(self.yaw - first_angle) < 0.25:
                                self.yaw_calibrate = True
                                yaw_count = 1.
                                break
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

                if self.tran_flag and forearm_list is not None:
                    forearm_list = self._get_distance(torch.from_numpy(forearm_list))
                    forearm_coordinate = self._transform_kp(forearm_list)
                    center = forearm_coordinate[0, 0] if forearm_coordinate[0, 0, 2] > forearm_coordinate[0, 1, 2] else forearm_coordinate[0, 1]
                    if (forearm_coordinate[0, 0, :] == forearm_coordinate[0, 1, :]).all():
                        forearm_coordinate = forearm_coordinate[1:, :, :]
                    vector_diff = forearm_coordinate[:, 0, :] - forearm_coordinate[:, 1, :]
                    forearm_roll_list = torch.rad2deg(torch.atan(vector_diff[:, 2] / vector_diff[:, 0]))
                    forearm_roll = torch.mean(torch.sort(forearm_roll_list).values)
                    roll_sin = torch.sin(torch.deg2rad(forearm_roll))
                    roll_cos = torch.cos(torch.deg2rad(forearm_roll))
                    A_apriltag = center + torch.matmul(
                        self.center2A, torch.tensor([[roll_sin, 0, roll_cos], [0, 1, 0], [roll_cos, 0, -roll_sin]], dtype=torch.float64))
                    Adown_apriltag = A_apriltag + torch.tensor(
                        [roll_cos * self.A2Adown, 0, roll_sin * self.A2Adown], dtype=torch.float)

                    self.apriltag_bag = torch.cat((torch.tensor(rospy.get_rostime().nsecs).unsqueeze(0), 
                                                   forearm_roll.unsqueeze(0), A_apriltag, Adown_apriltag))
                forearm_receive.clear()

    def accel_process(self, accel_receive, calibrate_array, calibrate_array_size):
        '''
        这进程负责获取加速度计信息并使用递归算法获取当前值用于姿态校正
        获取的当前值有较大迟滞性。

        @param accel_receive: 事件被设置加速度计数据持续更新

        @param calibrate_array: 最新加速度计数据数组
        @type  name: multiprocessing.Array

        @param calibrate_array_size: 当前加速度计数据被更新次数
        @type  name: multiprocessing.Value

        '''
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
        while True:
            try:
                    pipeline.start(config)#等待直至相机连接
                    break
            except Exception as e:
                rospy.logwarn_throttle_identical(60,"Please connect camera.")
        while not rospy.is_shutdown():
        # 接收到加速度信息更新加速度数据
            while accel_receive.is_set():
                try:
                    frames = pipeline.wait_for_frames()
                    accel_frame = frames.first_or_default(rs.stream.accel)
                    if accel_frame:
                        accel_data = accel_frame.as_motion_frame().get_motion_data()
                        xyz = [accel_data.x, accel_data.y, accel_data.z]
                        if calibrate_array_size.value > 0:
                            # 使用递归算法更新calibrate_array
                            calibrate_array[0] = calibrate_array[0] + \
                                (xyz[0] - calibrate_array[0]) / calibrate_array_size.value
                            calibrate_array[1] = calibrate_array[1] + \
                                (xyz[1] - calibrate_array[1]) / calibrate_array_size.value
                            calibrate_array[2] = calibrate_array[2] + \
                                (xyz[2] - calibrate_array[2]) / calibrate_array_size.value
                            calibrate_array_size.value += 1
                        else:
                            calibrate_array[0] = xyz[0]
                            calibrate_array[1] = xyz[1]
                            calibrate_array[2] = xyz[2]
                            calibrate_array_size.value = 1
                except Exception as e:
                    rospy.logwarn_throttle_identical(60,
                                                    "The accelerometer is subject to severe vibration. Please reconnect the camera.")
        pipeline.stop()

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
                return self._get_distance(keypoints)
            return None

    def _transform_kp(self, kp_tensor): 
        '''
        通过(彩色相机->机体)旋转矩阵变换关键点坐标.

        @param kp_tensor: 包含关键点三维坐标的张量, 形状为 (n, m, 3).
                        'n' 代表不同种类的数量, 'm' 代表同一种类中关键点的数量.
        @type kp_tensor: torch.Tensor

        @return: 变换后的关键点坐标张量, 形状与输入相同.
        @rtype: torch.Tensor 或 None

        如果输入的 kp_tensor 不为空, 此函数将对每个关键点应用旋转变换, 
        并返回变换后的坐标张量. 如果输入为空, 则返回 None. 
        '''
        if kp_tensor is not None:
            transform_tensor = torch.zeros(kp_tensor.size(0), kp_tensor.size(1), 3, dtype=torch.float64)
            for row in range(kp_tensor.size(0)):
                for col in range(kp_tensor.size(1)):
                    point = torch.tensor([[kp_tensor[row][col][0]], [kp_tensor[row][col][1]], [1.]])
                    point = torch.matmul(kp_tensor[row][col][2] * torch.from_numpy(self.inv_K) , point)
                    if self.tran_flag:
                        point = torch.matmul(torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float64), point)
                        transform_tensor[row][col] = torch.matmul(point.squeeze(), self.R_color_to_body_tensor).squeeze()
                    else:
                         transform_tensor[row][col] = torch.matmul(
                             torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float64), point).squeeze()
                #x,y,z 前 右 下 tensor float64
            return transform_tensor
        return None

    def _bucket_attitude(self, AB_kps):
        '''
        计算铲斗的方位角, 并结合小臂的方位角输出

        @param AB_kps:包含A、B关键点的(2, m, 3)张量, 其中m为相同种类点的个数
        @type AB_kps:torch.Tensor
        
        @return: 返回一个包含小臂方位角和铲斗方位角的元组
        @rtype: tuple(torch.Tensor) 或 None

        此函数首先检查是否有有效的 AprilTag 数据包
        如果没有, 将记录警告并返回 None
        如果有新的 AprilTag 更新, 将计算并返回小臂和铲斗的方位角
        '''
        if self.apriltag_bag is not None:
            apriltag_msg = self.apriltag_bag
        else:
            rospy.logwarn_throttle_identical(5, "No apriltag bag.")
            return None
        if (self.forearm_thread_record_time is None or 
            self.forearm_thread_record_time != apriltag_msg[0]):
            self.forearm_thread_record_time = apriltag_msg[0]
        else:
            rospy.logwarn_throttle_identical(5, "No new aprilatg updates.")
            return None
        point_A = apriltag_msg[2:5]
        point_A_down = apriltag_msg[5:]
        point_B = self._correction_point_B(point_A, AB_kps)
        point_B_down = self._calculate_opposite_quadrilateral_vertex(point_A_down, point_B)
        A2B_down_distance = torch.norm(point_A - point_B_down)
        angle_NQK = torch.rad2deg(torch.arccos((
            self.quadrilateral_length[0]**2 + self.quadrilateral_length[3]**2 - A2B_down_distance)/
            (2 * self.quadrilateral_length[0] * self.quadrilateral_length[3])))
        bucket_roll = angle_NQK - self.apriltag_bag[1] + 5.5 + 108
        return self.apriltag_bag[1], bucket_roll
        
    def _calculate_opposite_quadrilateral_vertex(self, vertex_A, vertex_B):
        '''
        计算由两个圆心和半径定义的两个圆的交点，这两个交点代表四边形的对角顶点

        @param vertex_A: 第一个圆的圆心坐标
        @param vertex_B: 第二个圆的圆心坐标
        @type vertex_A, vertex_B: torch.Tensor

        @return: 位于更高z坐标的交点, 代表四边形的对角顶点
        @rtype: torch.Tensor

        此函数通过计算两个圆的交点来确定四边形的一个未知顶点
        交点是通过解析几何方法得出的
        '''
        radius_A = self.quadrilateral_length[3]
        radius_B = self.quadrilateral_length[2]
        center_distance = torch.norm(vertex_B - vertex_A)
        a = (radius_A**2 - radius_B**2 + center_distance**2) / (2 * center_distance)
        h = torch.sqrt(radius_A**2 - a**2)
        midpoint = vertex_A + a * (vertex_B - vertex_A) / center_distance
        intersection1 = midpoint + h * torch.tensor(
            [-1 * (vertex_B[2] - vertex_A[2]) / center_distance, 0, (vertex_B[0] - vertex_A[0]) / center_distance], dtype=torch.float64)
        intersection2 = midpoint - h * torch.tensor(
            [-1 * (vertex_B[2] - vertex_A[2]) / center_distance, 0, (vertex_B[0] - vertex_A[0]) / center_distance], dtype=torch.float64)
        return intersection1 if intersection1[2] > intersection2[2] else intersection2

    def _correction_point_B(self, A_apriltag, AB_yolo_kps):
        '''
        通过apriltag_A与yolo推理A偏差来纠正yolo推理B, 并将B与apriltag_A对齐同一y平面

        @param A_apriltag: apriltag检测得到的A点的三维坐标。
        @param AB_yolo_kps: YOLO检测得到的A、B两点的三维坐标张量, 形状为 (2, m, 3), 其中m为同类点的数量
        @type A_apriltag, AB_yolo_kps: torch.Tensor

        @return: 校正后的B点三维信心
        @rtype: torch.Tensor

        该函数首先计算A点的y坐标与YOLO关键点y坐标的比例, 以此来调整AB_yolo_kps张量,
        使得B点的y坐标与A点对齐. 然后, 根据A点和B点的z坐标差异, 进一步调整B点的z坐标,
        以确保B点的准确性
        '''
        ac = (A_apriltag[1] / AB_yolo_kps[:, :, 1]).unsqueeze(1) * AB_yolo_kps - A_apriltag
        ac_normalized = ac / torch.norm(ac, dim=2, keepdim=True)
        AB_yolo_kps = A_apriltag + self.A2B * ac_normalized.squeeze(2)
        A_yolo, B_yolo = torch.mean(AB_yolo_kps, dim=1)
        A_delta_z = torch.clamp(A_apriltag[2] - A_yolo[2], -self.B_error/2., self.B_error/2.)
        B_yolo[2] += A_delta_z
        ab_vector = B_yolo - A_apriltag
        ab_vector_normalized = ab_vector / torch.norm(ab_vector)
        B_yolo_new = A_apriltag + self.A2B * ab_vector_normalized
        return B_yolo_new
            
    def _get_distance(self, kp_tensor):
        '''
        获取关键点距离数值并拼接进输入张量.

        @param kp_tensor: 包含关键点三维坐标的张量, 形状为 (n, m, 2).
                        'n' 代表不同种类的数量, 'm' 代表同一种类中关键点的数量.
        @type kp_tensor: torch.Tensor

        @return: 获得距离的关键点坐标张量, 形状为 (n, m, 3).
        @rtype: torch.Tensor
        '''
        add_distance_tensor = torch.zeros(kp_tensor.size(0), kp_tensor.size(1), 3, dtype=torch.float64)
        for row in range(kp_tensor.size(0)):
            for col in range(kp_tensor.size(1)):
                mid_pos = kp_tensor[row][col]
                distance = self._get_point_distance(mid_pos.int())
                add_distance_tensor[row][col] = torch.cat((mid_pos, distance))
        return add_distance_tensor

    def _get_point_distance(self, pixel_coordinates):
        '''
        计算指定像素坐标处关键点的距离, 单位为厘米, 原始深度图像数据单位为毫米

        @param pixel_coordinates: 指定关键点的像素坐标
        @type pixel_coordinates: torch.Tensor

        @return: 计算得到的关键点距离
        @rtype: torch.Tensor

        此函数在深度图像中搜索指定像素坐标周围的非零距离值,
        并计算这些值的统计量(如四分位数)来估算关键点的距离
        如果搜索范围内没有非零值, 则返回默认距离
        '''
        distance_list = torch.tensor([])
        for scope in range(self.point_search_range):
            if (pixel_coordinates[0] + scope < self.depth_image.shape[1] and 
                pixel_coordinates[1] + scope < self.depth_image.shape[0]):
                distance_list = self.depth_image[
                    pixel_coordinates[1]-scope:pixel_coordinates[1]+scope+1,
                    pixel_coordinates[0]-scope:pixel_coordinates[0]+scope+1
                ].flatten()
                if torch.sum(distance_list) != 0:
                    break
        distance_list = distance_list[distance_list != 0]
        if len(distance_list) >= 4:
            distance_list = torch.quantile(distance_list, torch.tensor([0.25, 0.5, 0.75]))
        elif len(distance_list) == 0:
            return torch.tensor(10.).unsqueeze(0)
        return (torch.round(torch.mean(distance_list)* 1000) / 10000).unsqueeze(0)
        
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
        yaw = self.yaw
        rospy.loginfo("Calibration successful!")
        rospy.loginfo("pitch: %2f, roll: %2f, yaw: %2f", pitch, self.roll, yaw)

        # 计算并发布(彩色相机->机体)旋转矩阵
        color_to_body_rotation = R.from_euler('XYZ', [pitch, self.roll, yaw], degrees=True)
        self.R_color_to_body_tensor = torch.tensor(color_to_body_rotation.as_matrix(), dtype=torch.float64)
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

    def _time_test(self, key=1):
        '''
        测试指定函数段耗时辅助函数, 并通过info等级消息输出结果

        @param key:1为开始计时, 2为结束计时
        @type: int
        '''
        if key == 1:
            self.start_time = rospy.get_rostime()
        if key == 2:
            time1 = (rospy.get_rostime().secs - self.start_time.secs)*1000 + (rospy.get_rostime().nsecs - self.start_time.nsecs)/1000000
            rospy.loginfo(time1, "ms")

 
def main():
    debug = rospy.get_param('~debug', 'False') == True
    if debug:
        rospy.init_node('cv_joint_angle', log_level=rospy.DEBUG, anonymous=True)
    else:
        rospy.init_node('cv_joint_angle', log_level=rospy.INFO, anonymous=True)
    cv_joint_angle = TargetDetection()
    rospy.spin()


if __name__ == "__main__":
    main()
