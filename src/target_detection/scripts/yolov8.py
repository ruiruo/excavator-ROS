#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import math
from scipy.spatial.transform import Rotation as R
import scipy.constants as C
import threading
import multiprocessing
import apriltag
import torch

import rospy
from geometry2.tf2_ros.src import tf2_ros
from geometry_msgs.msg import TransformStamped


class TargetDetection:
    def __init__(self):

        # 加载参数
        weight_path = rospy.get_param('~weight_path', 
                                      "$(find target_detection)/weights/yolov8n-pose.onnx")
        pub_topic = rospy.get_param('~pub_topic', '/coordinate_transform/CoordinatePoint')
        self.half = rospy.get_param('~half', 'False')
        self.show = rospy.get_param('~show', 'False')# 显示实时画面
        self.conf = rospy.get_param('~conf', '0.5')
        self.point_search_range = int(rospy.get_param('~point_search_range', '4'))
        self.model = YOLO(weight_path, task='pose')
        self.debug = rospy.get_param('~debug', 'False')
        self.target_box = rospy.get_param('~target_box', 'False')
        torch.set_printoptions(sci_mode=False)

        # 选择yolo运行设备
        if (rospy.get_param('~cpu', 'False')):
            self.device = 'cpu'
        else:
            self.device = '0'

        self.cam_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.color_cam_to_body_tf = TransformStamped()
        self.yaw_calibrate = False
        self.seq = 0
        # 彩色相机内参按照图片旋转90度变换
        self.K = np.array([[622.461673, 0., 225.527207], 
                           [  0., 620.817214, 418.324731],[  0., 0., 1., ]], dtype=np.float64)
        self.inv_K = np.linalg.inv(self.K) 
        self.tran_flag = False
        self.accel_calibrate_array = multiprocessing.Array('f', 3)
        self.accel_calibrate_array_size = multiprocessing.Value('i', 0)

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
        align_to = rs.stream.color
        align = rs.align(align_to)

        # 小臂apriltag配置
        self.forearm_detector = apriltag.Detector(apriltag.DetectorOptions(
            families="tag36h11", refine_pose=True))#开启姿态解算优化后旋转矩阵默认旋转顺序XYZ

        # tf坐标变换话题
        self.conversion_buffer = tf2_ros.Buffer(rospy.Time())
        self.conversion_listener = tf2_ros.TransformListener(self.conversion_buffer)

        #发布检测框与关键点话题
        # self.position_pub = rospy.Publisher(pub_topic, Coordinate_points, queue_size=10)

        # 子线程用于接受加速度计数据
        accel_receive = multiprocessing.Event()
        accel_receive.set()
        accel_p = multiprocessing.Process(target=self.accel_process, \
                                          args=(accel_receive, self.accel_calibrate_array, self.accel_calibrate_array_size,))
        accel_p.start()
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
                    # self.coordinate_points.header.stamp = rospy.Time.now()
                    # self.seq =+ 1
                    # self.coordinate_points.header.seq = self.seq
                    # self.coordinate_points.header.frame_id = "camera_color_frame"

                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    self.color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
                    depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
                    depth_image = depth_image.astype(np.float32)
                    self.depth_image = torch.from_numpy(depth_image)
                    forearm_receive.set()
                    predict_kp = self._predict()

                # 将校正后得到的相机外参用于关键点转换
                if self.accel_calibrate_array_size.value >= 250 and self.yaw_calibrate and not self.tran_flag:
                    accel_receive.clear()
                    self._calibrate()
                    self.accel_calibrate_array = multiprocessing.Array('f', 3)
                    self.accel_calibrate_array_size = multiprocessing.Value('i', 0)

                # 使用相机外参对关键点转换并发布 耗时2-4ms
                if self.tran_flag:
                    self._transform_kp(predict_kp)
            # except Exception as e:
            #     rospy.logwarn_throttle_identical(60,\
            #                                      "Unable to obtain camera image data, check if camera is working.")
        accel_p.join()
        pipeline.stop()
        cv2.destroyAllWindows()

    def _predict(self):
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
                    keypoints = results[0].keypoints.xy
                return self._get_distance(keypoints)
            return None

    def forearm_thread(self, forearm_receive):
        yaw_count = 1.
        while not rospy.is_shutdown():
            while forearm_receive.is_set():
                yuv_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2YUV)
                yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
                gray = yuv_image[:, :, 0]
                tags = self.forearm_detector.detect(gray)
                forearm_list = None
                for tag in tags:
                    if tag.tag_family == b'tag36h11' and tag.tag_id == 5:
                        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(tag.homography, self.K)
                        if not self.tran_flag:
                            angle = [-np.degrees(np.arctan(r[1,0]/r[0,0])) for r in Rs[::2]]
                            if math.isclose(yaw_count, 1.):
                                self.yaw = angle[0]
                            else:
                                yaw = self.yaw + (angle[0] - self.yaw)/yaw_count
                                if abs(self.yaw - yaw) < 0.25 and 10<self.yaw<30:
                                    self.yaw_calibrate = True
                                self.yaw = yaw
                            yaw_count += 1
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

                if self.tran_flag:
                    if (forearm_list[0,0,:] == forearm_list[0,1,:]).all():
                        forearm_list = forearm_list[1:,:,:]
                    forearm_list = self._get_distance(torch.from_numpy(forearm_list))
                    forearm_coordinate = self._transform_kp(forearm_list)
                    forearm_roll_list = torch.tensor(
                        [torch.rad2deg(torch.atan(vector[2]/vector[0])).item() 
                         for vector in (row[0] - row[1] for row in forearm_coordinate)])
                    self.forearm_roll = torch.mean(torch.sort(forearm_roll_list).values[1:-1])
                forearm_receive.clear()

    def accel_process(self, accel_receive, calibrate_array, calibrate_array_size):
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

    # 将关键点坐标通过(彩色相机->机体)旋转矩阵进行变换
    def _transform_kp(self, kp_tensor): 
        if kp_tensor is not None:
            transform_tensor = torch.zeros(kp_tensor.size(0), kp_tensor.size(1), 3, dtype=float)
            for row in range(kp_tensor.size(0)):
                for col in range(kp_tensor.size(1)):
                    point = torch.tensor([[kp_tensor[row][col][0]], [kp_tensor[row][col][1]], [1.]])
                    point = torch.matmul(kp_tensor[row][col][2] * torch.from_numpy(self.inv_K) , point)
                    if self.tran_flag:
                        point = torch.matmul(torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float), point)
                        transform_tensor[row][col] = torch.matmul(point.squeeze(), self.R_color_to_body_tensor).squeeze()
                    else:
                         transform_tensor[row][col] = torch.matmul(
                             torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float), point).squeeze()
                #x,y,z 前 右 下 tensor float64
            # print(transform_tensor)
            return transform_tensor
        return None

    def _get_distance(self, kp_tensor):
        add_distance_tensor = torch.zeros(kp_tensor.size(0), kp_tensor.size(1), 3, dtype=float)
        for row in range(kp_tensor.size(0)):
            for col in range(kp_tensor.size(1)):
                mid_pos = kp_tensor[row][col]
                distance = self._get_point_distance(mid_pos.int())
                add_distance_tensor[row][col] = torch.cat((mid_pos, distance))
        return add_distance_tensor

    # 获取指定关键点距离，距离单位cm 原始数据单位为mm
    def _get_point_distance(self, mid_pos):
        distance_list = torch.tensor([])
        for scope in range(self.point_search_range):
            if (mid_pos[0] + scope < self.depth_image.shape[1] and 
            mid_pos[1] + scope < self.depth_image.shape[0]):
                distance_list = self.depth_image[
                    int(mid_pos[1]-scope):int(mid_pos[1]+scope+1),
                    int(mid_pos[0]-scope):int(mid_pos[0]+scope+1)
                ].flatten()
            if torch.sum(distance_list) != 0:
                break
        distance_list = distance_list[distance_list != 0]
        if len(distance_list) >= 4:
            distance_list = torch.quantile(distance_list, torch.tensor([0.25, 0.5, 0.75]))
        elif len(distance_list) == 0:
            return torch.tensor(10.).unsqueeze(0)
        return (torch.round(torch.mean(distance_list)* 1000) / 10000).unsqueeze(0)
        
    # RGBD相机外参旋转矩阵获取
    def _calibrate(self):
        # 加速度计坐标到相机坐标系仅有平移关系
        aceel_to_color = R.from_quat([0.00121952651534, -0.00375633803196, 
                                         -0.000925257743802, 0.999991774559]).as_euler('XYZ', degrees=True)
        calibrate_angle = np.clip(self.accel_calibrate_array, -C.g, C.g) 
        roll = -90 - math.degrees(math.atan(calibrate_angle[0] /         #当前姿态与正下的差值 右为正方向
                                        math.sqrt(calibrate_angle[1]**2 + calibrate_angle[2]**2)
                                        if not math.isclose(calibrate_angle[1]**2 + calibrate_angle[2]**2, 0.) else float('inf')))
        pitch = math.degrees(math.atan(calibrate_angle[1] /               #当前姿态与正前的差值 前为正方向
                                       math.sqrt(calibrate_angle[0]**2 + calibrate_angle[2]**2)
                               if not math.isclose(calibrate_angle[0]**2 + calibrate_angle[2]**2, 0.) else float('inf')))
        #缺少body与world的旋转矩阵 yaw直接参考机体 roll与pitch参考世界坐标系
        self.roll = roll + aceel_to_color[1]
        pitch = pitch + aceel_to_color[0]
        yaw = self.yaw
        rospy.loginfo("Calibration successful! ")
        rospy.loginfo("pitch:%2f roll: %2f  yaw:  %2f", pitch , self.roll,  yaw)

        # (彩色相机->机体)旋转矩阵计算与发布
        R_color_to_body = R.from_euler('XYZ', [pitch, self.roll, yaw], degrees=True)
        self.R_color_to_body_tensor = torch.tensor(R_color_to_body.as_matrix(), dtype=torch.float64)

        self.color_cam_to_body_tf.header.frame_id = "camera_color_frame"
        self.color_cam_to_body_tf.header.stamp = rospy.Time.now()
        self.color_cam_to_body_tf.header.seq = 0
        self.color_cam_to_body_tf.child_frame_id = "body"
        self.color_cam_to_body_tf.transform.translation.x = 0.
        self.color_cam_to_body_tf.transform.translation.y = 0.
        self.color_cam_to_body_tf.transform.translation.z = 0.
        self.color_cam_to_body_tf.transform.rotation.x = R_color_to_body.as_quat()[0]
        self.color_cam_to_body_tf.transform.rotation.y = R_color_to_body.as_quat()[1]
        self.color_cam_to_body_tf.transform.rotation.z = R_color_to_body.as_quat()[2]
        self.color_cam_to_body_tf.transform.rotation.w = R_color_to_body.as_quat()[3]
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
