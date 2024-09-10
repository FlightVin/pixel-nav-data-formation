import os
import numpy as np
import habitat
from habitat.datasets.image_nav.instance_image_nav_dataset import InstanceImageNavDatasetV1
from habitat.tasks.nav.instance_image_nav_task import InstanceImageNavigationTask
from pprint import pprint
from habitat.config.read_write import read_write
from tqdm import tqdm
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import cv2 
import quaternion
import open3d as o3d
import time

def habitat_camera_intrinsic(config):
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov, 'The configuration of the depth camera should be the same as rgb camera.'
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f,0,xc],
                                 [0,f,zc],
                                 [0,0,1]],np.float32)
    return intrinsic_matrix

def get_pointcloud_from_depth(rgb:np.ndarray,depth:np.ndarray,intrinsic:np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z,filter_x = np.where(depth>-1)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z,filter_x]
    point_values = np.stack([pixel_x,pixel_z,-pixel_y],axis=-1)
    return filter_x,filter_z,point_values,color_values

def translate_to_world(points:np.ndarray,position:np.ndarray,rotation:np.ndarray):
    extrinsic = np.eye(4)
    extrinsic[0:3,0:3] = rotation 
    extrinsic[0:3,3] = position
    world_points = np.matmul(extrinsic,np.concatenate((points,np.ones((points.shape[0],1))),axis=-1).T).T
    return world_points[:,0:3]

def random_pixel_goal(habitat_config,habitat_env, difficulty='medium'):
    camera_int = habitat_camera_intrinsic(habitat_config)
    robot_pos = habitat_env.sim.get_agent_state().position
    robot_rot = habitat_env.sim.get_agent_state().rotation
    camera_pos = habitat_env.sim.get_agent_state().sensor_states['rgb'].position
    camera_rot = habitat_env.sim.get_agent_state().sensor_states['rgb'].rotation
    camera_obs = habitat_env.sim.get_observations_at(robot_pos,robot_rot)
    rgb = camera_obs['rgb']
    depth = camera_obs['depth']
    xs, zs, rgb_points, rgb_colors = get_pointcloud_from_depth(rgb, depth, camera_int)
    rgb_points = translate_to_world(rgb_points,camera_pos,quaternion.as_rotation_matrix(camera_rot))
    condition_index = np.where((rgb_points[:,1] < robot_pos[1] + 1.0) & (rgb_points[:,1] > robot_pos[1] - 0.2) & (depth[(zs,xs)][:,0] > 3.0) & (depth[(zs,xs)][:,0] < 5.0))[0]
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(rgb_points[condition_index])
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_colors[condition_index]/255.0)
    if condition_index.shape[0] == 0:
        return False, [], [], []
    else:
        random_index = np.random.choice(condition_index)
        target_x = xs[random_index]
        target_z = zs[random_index]
        target_point = rgb_points[random_index]
        min_z = max(target_z-5,0)
        max_z = min(target_z+5,depth.shape[0])
        min_x = max(target_x-5,0)
        max_x = min(target_x+5,depth.shape[1])
        target_mask = np.zeros((depth.shape[0],depth.shape[1]),np.uint8)
        target_mask[min_z:max_z,min_x:max_x] = 1
        target_point[1] = robot_pos[1]
        return True, rgb, target_mask, target_point

# Define the stage
stage = "train"
split = "train"
num_sampled_episodes = 1000
num_saved_episodes = int(1e4)
max_timesteps = 64
min_timesteps = 5
save_path = "/scratch/vineeth.bhat/pix_nav_point_based_data/training"

config_path = "hm3d_config_instance_image_nav_mod.yaml"

if not os.path.exists(config_path):
    raise RuntimeError(f"{config_path} does not exist!")

habitat_config = habitat.get_config(config_path)

data_path = f"/scratch/vineeth.bhat/sg_habitat/data/datasets/instance_imagenav/hm3d/v3/{stage}/{stage}.json.gz"
scene_dataset = f"/scratch/vineeth.bhat/sg_habitat/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
scenes_dir = "/scratch/vineeth.bhat/sg_habitat/data/scene_datasets"

if not os.path.exists(data_path):
    raise RuntimeError(f"{data_path} does not exist!")

if not os.path.exists(scene_dataset):
    raise RuntimeError(f"{scene_dataset} does not exist!")

# Update habitat configuration
robot_height=0.88
robot_radius=0.25
sensor_height=1.31
image_width=224
image_height=224
image_hfov=79
step_size=0.25
turn_angle=30
with read_write(habitat_config):
    habitat_config.habitat.dataset.split = split
    habitat_config.habitat.dataset.scenes_dir = scenes_dir
    habitat_config.habitat.dataset.data_path = data_path
    habitat_config.habitat.simulator.scene_dataset = scene_dataset
    habitat_config.habitat.environment.iterator_options.num_episode_sample = num_sampled_episodes
    habitat_config.habitat.simulator.agents.main_agent.height=robot_height
    habitat_config.habitat.simulator.agents.main_agent.radius=robot_radius
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = image_height
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = image_width
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = image_hfov
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0,sensor_height,0]
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = image_height
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = image_width
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = image_hfov
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0,sensor_height,0]
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 50.0
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = 0.0
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False

    habitat_config.habitat.simulator.forward_step_size = step_size
    habitat_config.habitat.simulator.turn_angle = turn_angle

pprint(habitat_config.habitat.simulator.agents.main_agent.sim_sensors)

try:
    env = habitat.Env(habitat_config)
    print("Environment initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize the Habitat environment: {e}")

episode_counter = 0

follower = ShortestPathFollower(env.sim, goal_radius=1.5, return_one_hot=False)

time_before_loop = time.time()

while episode_counter < num_saved_episodes:
    start_time = time.time()
    obs = env.reset()
    rgb_data = []
    depth_data = []
    pose_data = []
    action_data = []
    timesteps = 0
    # goal_position = env.current_episode.goals[0].position

    goal_flag, goal_image, goal_mask, goal_point = random_pixel_goal(habitat_config, env)

    if goal_flag == False:
        print(f"Rejected the current goal {goal_flag}")
        continue

    start_rgb_image = obs["rgb"]
    start_depth_image = obs["depth"]
    
    # Iterate over every step of the episode
    while True:
        best_action = follower.get_next_action(goal_point)
        if best_action is None:
            print(f"Goal reached at timestep {timesteps}")
            break

        action_data.append(best_action)
        obs = env.step(best_action)
        
        rgb_data.append(obs['rgb'])
        depth_data.append(obs['depth'])

        q = env.sim.get_agent_state().sensor_states["depth"].rotation
        pose = np.concatenate(
            [np.array(env.sim.get_agent_state().sensor_states["depth"].position),
            np.array([q.w, q.x, q.y, q.z])]
        )

        pose_data.append(pose)

        timesteps += 1

        if env.episode_over or timesteps >= max_timesteps:
            if timesteps < min_timesteps:
                print(f"{timesteps} less than min. timesteps of {min_timesteps}")
                break

            episode_save_path = os.path.join(save_path, f"episode_{episode_counter}")
            os.makedirs(episode_save_path)

            # Save initial data
            cv2.imwrite(os.path.join(episode_save_path, "start_rgb_image.png"), start_rgb_image)
            np.save(os.path.join(episode_save_path, "start_depth_image.npy"), start_depth_image)
            np.save(os.path.join(episode_save_path, "goal_mask.npy"), goal_mask)

            start_image_with_mask = start_rgb_image.copy()
            goal_mask_color = np.stack([goal_mask]*3, axis=-1)
            color_overlay = np.array([0, 0, 255], dtype=np.uint8)
            start_image_with_mask[goal_mask == 1] = color_overlay
            cv2.imwrite(os.path.join(episode_save_path, "start_rgb_image_with_mask.png"), start_image_with_mask)
                        
            # save trajectory data
            pose_data_path = os.path.join(episode_save_path, "poses.npy")
            np.save(pose_data_path, pose_data)

            action_data_path = os.path.join(episode_save_path, "actions.npy")
            np.save(action_data_path, action_data)

            rgb_images_save_path = os.path.join(episode_save_path, "rgb")
            depth_images_save_path = os.path.join(episode_save_path, "depth")
            os.makedirs(rgb_images_save_path)
            os.makedirs(depth_images_save_path)

            for i in range(len(rgb_data)):
                rgb_image_path = os.path.join(rgb_images_save_path, f"{i}.png")
                depth_image_path = os.path.join(depth_images_save_path, f"{i}.npy")

                cv2.imwrite(rgb_image_path, rgb_data[i])

                # Save depth data
                np.save(depth_image_path, depth_data[i])

            episode_counter += 1
            end_time = time.time()
            time_taken = end_time - start_time 
            total_time_taken = end_time - time_before_loop
            total_hours, rem = divmod(total_time_taken, 3600)
            total_minutes, total_seconds = divmod(rem, 60)

            print(f"Done with episode {episode_counter} out of {num_saved_episodes} total")
            print(f"Time taken for this episode: {time_taken:.2f} seconds")
            print(f"Total time taken: {int(total_hours)} hours, {int(total_minutes)} minutes, {int(total_seconds)} seconds")
            break

env.close()