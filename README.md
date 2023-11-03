# Yolov5_ros

提供了一个基于[PyTorch-YOLOv5](https://github.com/ultralytics/yolov5/tree/v6.2)的ROS功能包。该功能包已在Ubuntu 18.04上进行了测试。

## 开发环境

- Ubuntu 18.04
- [ROS melodic](http://wiki.ros.org/cn/melodic/Installation/Ubuntu)
- Python == 3.6.9环境，PyTorch == 1.10.0([Jetson TX2安装方法](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048))

## 代码下载

- `git clone https://gitee.com/lisq58/excavator_detect.git`
- `git submodule update --init --recursive`
- 检查[src/realsense-ros]( https://gitee.com/lisq58/excavator_detect/tree/master/src) , [src/rgbd_launch](https://gitee.com/lisq58/excavator_detect/tree/master/src), [src/yolov5_ros/yolov5_ros/yolov5](https://gitee.com/lisq58/excavator_detect/tree/master/src/yolov5_ros/yolov5_ros)目录下子模块均有内容即可

## 基本用法

1. 首先，确保将训练好的权重放在 [weights 文件夹](https://gitee.com/lisq58/excavator_detect/src/yolov5_ros/yolov5_ros/weights)中。
2. [yolo_v5.launch文件](https://gitee.com/lisq58/excavator_detect/src/yolov5_ros/yolov5_ros/launch/yolo_v5.launch) 中设置权重文件，另外需要在launch文件中额外修改您对应的摄像头话题名称以及是否使用Cpu选项
3. 使用该功能包

- 打开realsense的相机节点

```
roslaunch realsense_carmera rs_camera.launch
```

- 打开yolov5图像识别节点

```
roslaunch yolov5_ros yolo_v5.launch
```
