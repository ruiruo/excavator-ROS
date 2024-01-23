#!/bin/bash
# -*- coding: UTF-8 -*-

init(){
    echo -e "All nodes start! "
    gnome-terminal --window -- roslaunch realsense2_camera rs_rgbd.launch color_width:=848 color_height:=480 depth_width:=848 depth_height:=480 color_fps:=30 depth_fps:=30
    sleep 2
    gnome-terminal --window -- bash -c "source ~/workspace/yolov8_venv/bin/activate; roslaunch target_detection yolov8.launch" 
    sleep 2
    gnome-terminal --window -- bash -c "source ~/workspace/yolov8_venv/bin/activate; roslaunch coordinate_transform conversion.launch"
    echo -e "All nodes are ready! "
    sleep 8
}

runing(){
    key=""
    # 定义一个循环，无限执行，直到按下任意键
    while [ -z "$key" ]; do
        # 用tail和grep命令判断文件的最后一行是否以指定的字符串结尾
        tail -n 1 ~/.ros/log/latest/target_detection* | grep -q "waiting for image form target detect node.$"
        status1=$?
        tail -n 1 ~/.ros/log/latest/coordinate* | grep -q "waiting for accel info form realsense node.$"
        status2=$?
        if [ $status1 -eq 0 ] || [ $status2 -eq 0 ]; then
            echo -e "Restart  realsense2_camera..."
            rosservice call /camera/realsense2_camera/reset
        else
            echo -e "Runing... "
        fi
        # 用read命令读取标准输入，-t选项表示超时时间，-n选项表示字符数，-s选项表示静默模式
        # 如果在10秒内输入了一个字符，那么赋值给key变量，否则赋值为空
        read -p "Press any key to exit..." -t 20 -n 1 -s key 
        if [ -z "$key" ]; then
            echo " "
        fi
        trap key=1 SIGINT SIGTERM
    done
    echo " "
}

exit(){
    # 定义一个循环，无限执行，直到找不到有roslaunch的进程
    while true; do
        count=$(pgrep -f -c roslaunch)
        if [ $count -eq 0 ]; then
            break
        fi
        pkill -SIGINT -f roslaunch
        sleep 1
    done
    echo -e "Use 'rm -rf  ~/.ros/log' to trash log "
    echo -e "All nodes have been shut down!"
}

init
runing
exit
