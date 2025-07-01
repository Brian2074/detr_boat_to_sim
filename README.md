# DETR Boat to Sim
add launch file to huggingface-detr/ros1_ws/src/detr_inference/launch  

add spawn_model.py to huggingface-detr/ros1_ws/src/detr_inference/src  


# Another terminal

```bash
cd ~/robotx-2022  

source ipc_run.sh  

cd catkin_ws  

catkin build  

roslaunch js_gazeo empty.launch
```

# How to Train ResNet

```bash
python3 train.py --data_dir <dataset_directory> --batch_size <batch_size (default 32)> --learning_rate <learning_rate (default 0.001)> --num_epochs <num_epochs (default 50)>
```

* View Training Process
```bash
tensorboard --logdir=runs/yaw_regression
```
open http://localhost:6006 in your browser to view