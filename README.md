# DETR Boat to Sim
add launch file to huggingface-detr/ros1_ws/src/detr_inference/launch
add spawn_model.py to huggingface-detr/ros1_ws/src/detr_inference/src

# Another terminal
cd ~/robotx-2022
source ipc_run.sh
cd catkin_ws
catkin build
roslaunch js_gazeo empty.launch
