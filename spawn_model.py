#!/usr/bin/env python
import rospy
import time
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, Quaternion
from gazebo_msgs.srv import SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

MAX_MARKER_MODELS = 2
MAX_POINTCLOUD_MODELS = 5
MODEL_LIFETIME_SEC = 3.0

spawned_models = {}  # name -> {'time': t, 'type': 'marker'|'point'}
marker_positions = {}
pointcloud_points = []

FIXED_BOX_SIZE = [2.0, 5.0, 1.5]  # 固定大小 for type:1


def spawn_box(name, pose, size):
    pose.orientation = Quaternion(0, 0, 0, 1)
    Z_OFFSET = 1.0  # 可以改大一點試試
    pose.position.z = max(size[2] / 2.0, pose.position.z + Z_OFFSET)

    sdf = f"""
    <sdf version='1.6'>
    <model name='{name}'>
        <pose>{pose.position.x} {pose.position.y} {pose.position.z} 0 0 0</pose>
        <static>true</static>
        <link name='link'>
            <visual name='visual'>
                <geometry>
                    <box>
                        <size>{size[0]} {size[1]} {size[2]}</size>
                    </box>
                </geometry>
                <material>
                    <ambient>0 1 0 1</ambient>
                    <diffuse>0 1 0 1</diffuse>
                </material>
            </visual>
        </link>
    </model>
    </sdf>
    """
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model(name, sdf, "", pose, "map")
        spawned_models[name] = {'time': time.time(), 'type': 'marker'}
    except Exception as e:
        rospy.logwarn(f"[SPAWN] Failed box '{name}': {e}")


def update_model_position(name, pose):
    pose.orientation = Quaternion(0, 0, 0, 1)
    state = ModelState()
    state.model_name = name
    state.pose = pose
    state.reference_frame = "map"
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_state(state)
        spawned_models[name]['time'] = time.time()
    except Exception as e:
        rospy.logwarn(f"[UPDATE] Failed to move model '{name}': {e}")


def marker_callback(msg):
    global marker_positions
    marker_positions.clear()
    for marker in msg.markers:
        if marker.id == 0 :
            marker_positions[f"marker_{marker.id}"] = marker.pose


def pointcloud_callback(msg):
    global pointcloud_points
    pointcloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))[:MAX_POINTCLOUD_MODELS]


def main():
    rospy.init_node("spawn_and_move_models")
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    rospy.wait_for_service('/gazebo/set_model_state')

    rospy.Subscriber("/js/real_velodyne/matched_bboxes", MarkerArray, marker_callback)
    rospy.Subscriber("/js/real_velodyne/clustered_pointcloud", PointCloud2, pointcloud_callback)

    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        now = time.time()

        # 處理 marker 模型
        for name, pose in marker_positions.items():
            if name not in spawned_models:
                spawn_box(name, pose, FIXED_BOX_SIZE)
            else:
                update_model_position(name, pose)

        # 處理 pointcloud
        for i, pt in enumerate(pointcloud_points):
            name = f"pt_{i}"
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pt
            pose.orientation = Quaternion(0, 0, 0, 1)
            if name not in spawned_models:
                spawn_box(name, pose, [0.3, 0.3, 0.3])
            else:
                update_model_position(name, pose)

        rate.sleep()


if __name__ == '__main__':
    main()
