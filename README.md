# Excavator_detect

提供了一个基于[Ultralytics YOLOv8](https://docs.ultralytics.com/zh/)的ROS功能包。该功能包已在Ubuntu 18.04上进行了测试。

## 开发环境

- Ubuntu 18.04
- [ROS melodic](http://wiki.ros.org/cn/melodic/Installation/Ubuntu)
- Python3.8与2.7环境，PyTorch == 1.10.0([Jetson TX2安装方法](https://zhuanlan.zhihu.com/p/55509535), [Pytorch源代码](https://github.com/pytorch/pytorch/tree/v1.10.0), [torchvision](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048))

## 代码下载与配置

- 克隆代码及其子模块

```
git clone https://gitee.com/lisq58/excavator_detect.git
```

```
git submodule update --init --recursive
```

- 检查 [src/mavlink, src/mavros, src/geometry2, src/vicon_bridge]( https://gitee.com/lisq58/excavator_detect/tree/master/src)目录下子模块均有内容即可

- 安装mavros的依赖与地理列表数据集

```
sudo apt-get install ros-melodic-geographic-msgs ros-melodic-mavros ros-melodic-mavros-extras libgeographic-dev -y
sudo ./src/mavros/mavros/scripts/install_geographiclib_datasets.sh
```

- 安装realsense的依赖

```
sudo apt-get install ros-melodic-realsense2-camera ros-melodic-realsense2-description -y
```

- 修改`~/.bashrc`使得其他节点能调用源代码编译的tf2

```
echo "export PYTHONPATH=$PYTHONPATH:~/workspace/excavator_detect/src" >> ~/.bashrc
source ~/.bashrc
```

- 安装python3.8

```
sudo apt-get install python3.8 python3-pip -y
```

- 创建python3.8虚拟环境

```
sudo -H python3.8 -m pip install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip
```

```
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
source ~/.bashrc
```

- 使用virtualenv创建虚拟环境，virtualenvname为虚拟环境名称，-p表示母python路径，如/usr/bin/python3.8
, 创建成功后启动虚拟环境

```
virtualenv path/to/virtualenvname -p path/to/python3
source path/to/vitualenvname/bin/activate
```

- 在虚拟环境中安装依赖, 若权重文件使用.onnx类型则额外安装onnx onnxruntime

```
pip install ultralytics pyrealsense2 apriltag rospkg
```

- 安装编译ROS包的依赖

```
sudo apt-get install python-catkin-tools libxml2 libxml2-dev libxslt1.1 libxslt1-dev -y
```

1. 清除以往的编译文件

```
catkin clean
```

2. 安装编译的依赖(非虚拟环境)

```
python3.6 -m pip install empy==3.3.4 catkin_pkg futur lxml
sudo apt-get install python-pip
python2 -m pip install empy==3.3.4
```

3. 使用python3解释器编译在python3环境下运行的节点(不要在虚拟环境编译)

```
catkin build mavlink mavros geometry2 cv_joint_angle --cmake-args   -DCMAKE_BUILD_TYPE=Release  -DPYTHON_EXECUTABLE=/usr/bin/python3    -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

4. 编译其余节点

```
catkin build
```

#### 若在TX2中使用

- 将`/opt/ros/melodic/share/cv_bridge/cmake/`文件94行最后一个索引改为`/usr/include/opencv4`

```
sudo gedit /opt/ros/melodic/share/cv_bridge/cmake/cv_bridgeConfig.cmake
```

## 基本用法

1. 首先，确保将训练好的权重放在 [cv_joint_angle/weights 文件夹](https://gitee.com/lisq58/excavator_detect/src/cv_joint_angle/weights)中。
2. [yolov8.launch文件](https://gitee.com/lisq58/excavator_detect/src/cv_joint_angle/launch/yolov8.launch) 中设置权重文件与推理相关参数设置
3. 将环境变量加入bash

```
echo "source your/workspace/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

4. 使用该功能包`./your/workspace/excavator_detect/run.sh`
