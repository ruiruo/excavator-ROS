#!/bin/bash
# -*- coding: UTF-8 -*-

init(){
    echo -e "All nodes start! "
    gnome-terminal --window -- bash -c "source ~/workspace/yolov8_venv/bin/activate; roslaunch target_detection yolov8.launch" 
    sleep 8
}

runing(){
    echo -e "All nodes are ready! "
    key=""
    # 定义一个循环，无限执行，直到按下任意键
    while [ -z "$key" ]; do
        echo -e "Runing... "
        # 用read命令读取标准输入，-t选项表示超时时间，-n选项表示字符数，-s选项表示静默模式
        # 如果在10秒内输入了一个字符，那么赋值给key变量，否则赋值为空
        read -p "Press any key to exit..." -t 120 -n 1 -s key 
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
