# Excavator_detect

提供了一个基于[Ultralytics YOLOv8](https://docs.ultralytics.com/zh/)的ROS功能包。该功能包已在Ubuntu 18.04上进行了测试。

## 开发环境

- Ubuntu 18.04
- [ROS melodic](http://wiki.ros.org/cn/melodic/Installation/Ubuntu)
- Python3.8与2.7环境，PyTorch == 1.10.0([Jetson TX2安装方法](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048))

## 代码下载与配置

- 克隆代码及其子模块

```
git clone https://gitee.com/lisq58/excavator_detect.git
```

```
git submodule update --init --recursive
```

- 检查[src/realsense-ros]( https://gitee.com/lisq58/excavator_detect/tree/master/src) , [src/rgbd_launch](https://gitee.com/lisq58/excavator_detect/tree/master/src) , [src/mavlink]( https://gitee.com/lisq58/excavator_detect/tree/master/src) , [src/mavros]( https://gitee.com/lisq58/excavator_detect/tree/master/src) , [src/geometry2]( https://gitee.com/lisq58/excavator_detect/tree/master/src)目录下子模块均有内容即可
- 阅读[src/realsense-ros/README.md](https://gitee.com/lisq58/my_realsense_ros1/blob/my_realsense_ros1/README.md)安装`ros-melodic-realsense2-camera`与`ros-melodic-realsense2-description`

```
sudo apt-get install ros-melodic-realsense2-camera -y
```

```
sudo apt-get install ros-melodic-realsense2-description -y
```

- 安装mavros的依赖与地理列表数据集

```
rosdep install --from-paths src/mav* --ignore-src -y
./src/mavros/mavros/scripts/install_geographiclib_datasets.sh
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

- 在虚拟环境中安装ultralytics

```
pip install ultralytics
```

- 编译ROS包

1. 清除以往的编译文件

```
catkin clean
```

2. 安装编译的依赖(非虚拟环境编译)

```
python3.6 -m pip install empy==3.3.4 catkin_pkg 
sudo apt-get install python-pip
python2 -m pip insatll empy==3.3.4
```

3. 使用python3解释器编译在python3环境下运行的节点(不要在虚拟环境编译)

```
catkin build geometry2 coordinate_transform target_detection --cmake-args   -DCMAKE_BUILD_TYPE=Release  -DPYTHON_EXECUTABLE=/usr/bin/python3    -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

4. 按照realsense官方`README.md`编译realsense相关节点

```
catkin build realsense2_camera realsense2_description   -DCATKIN_ENABLE_TESTING=False   -DCMAKE_BUILD_TYPE=Release
```

5. 编译其余节点

```
catkin build
```

#### 若在TX2中使用

- 将`/opt/ros/melodic/share/cv_bridge/cmake/`文件94行最后一个索引改为`/usr/include/opencv4`

```
sudo gedit /opt/ros/melodic/share/cv_bridge/cmake/cv_bridgeConfig.cmake
```

## 基本用法

1. 首先，确保将训练好的权重放在 [target_detection 文件夹](https://gitee.com/lisq58/excavator_detect/src/target_detection)中。
2. [yolov8.launch文件](https://gitee.com/lisq58/excavator_detect/src/target_detection/launch/yolov8.launch) 中设置权重文件与推理相关参数设置
3. 将环境变量加入bash

```
echo "source your/workspace/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

4. 使用该功能包`./your/workspace/excavator_detect/run.sh`
