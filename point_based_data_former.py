import os
import argparse
import numpy as np
import habitat
from habitat.datasets.image_nav.instance_image_nav_dataset import InstanceImageNavDatasetV1
from habitat.tasks.nav.instance_image_nav_task import InstanceImageNavigationTask
from habitat.config.read_write import read_write
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import cv2 
import quaternion
import open3d as o3d
import time
from pathlib import Path
import h5py

def habitat_camera_intrinsic(config):
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f,0,xc],
                                 [0,f,zc],
                                 [0,0,1]], np.float32)
    return intrinsic_matrix

def get_pointcloud_from_depth(rgb, depth, intrinsic):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z, filter_x = np.where(depth > -1)
    depth_values = depth[filter_z, filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z, filter_x]
    point_values = np.stack([pixel_x, pixel_z, -pixel_y], axis=-1)
    return filter_x, filter_z, point_values, color_values

def translate_to_world(points, position, rotation):
    extrinsic = np.eye(4)
    extrinsic[0:3, 0:3] = rotation 
    extrinsic[0:3, 3] = position
    world_points = np.matmul(extrinsic, np.concatenate((points, np.ones((points.shape[0], 1))), axis=-1).T).T
    return np.array(world_points[:, 0:3])

def create_target_mask(target_x, target_z, mask_shape, depth_shape):
    min_z = max(target_z - mask_shape, 0)
    max_z = min(target_z + mask_shape, depth_shape[0])
    min_x = max(target_x - mask_shape, 0)
    max_x = min(target_x + mask_shape, depth_shape[1])
    target_mask = np.zeros((depth_shape[0], depth_shape[1]), np.uint8)
    target_mask[min_z:max_z,min_x:max_x] = 1
    return target_mask

def random_pixel_goal(habitat_config, habitat_env, mask_shape):
    camera_int = habitat_camera_intrinsic(habitat_config)
    robot_pos = habitat_env.sim.get_agent_state().position
    robot_rot = habitat_env.sim.get_agent_state().rotation
    camera_pos = habitat_env.sim.get_agent_state().sensor_states['rgb'].position
    camera_rot = habitat_env.sim.get_agent_state().sensor_states['rgb'].rotation
    camera_obs = habitat_env.sim.get_observations_at(robot_pos, robot_rot)
    rgb = camera_obs['rgb']
    depth = camera_obs['depth']
    xs, zs, rgb_points, rgb_colors = get_pointcloud_from_depth(rgb, depth, camera_int)
    rgb_points = translate_to_world(rgb_points, camera_pos, quaternion.as_rotation_matrix(camera_rot))
    condition_index = np.where((rgb_points[:,1] < robot_pos[1] + 1.0) & 
                               (rgb_points[:,1] > robot_pos[1] - 0.2) & 
                               (depth[(zs,xs)][:,0] > 1.0) & 
                               (depth[(zs,xs)][:,0] < 5.5))[0] # note - using different condition indices since different task dataset
    
    if condition_index.shape[0] == 0:
        return False, [], [], [], []
    else:
        random_index = np.random.choice(condition_index)
        target_x = xs[random_index]
        target_z = zs[random_index]
        target_point = rgb_points[random_index]
        target_mask = create_target_mask(target_x, target_z, mask_shape, depth.shape)
        original_target_point = target_point.copy()
        target_point[1] = robot_pos[1]
        return True, rgb, target_mask, target_point, original_target_point
    
def get_normalized_goal_point_location_in_current_obs(habitat_config, habitat_env, target_point):
    # https://github.com/wzcai99/Pixel-Navigator/issues/8#issuecomment-2378593390
    camera_int = habitat_camera_intrinsic(habitat_config)
    robot_pos = habitat_env.sim.get_agent_state().position
    robot_rot = habitat_env.sim.get_agent_state().rotation
    camera_pos = habitat_env.sim.get_agent_state().sensor_states['rgb'].position
    camera_rot = habitat_env.sim.get_agent_state().sensor_states['rgb'].rotation
    camera_obs = habitat_env.sim.get_observations_at(robot_pos,robot_rot)
    rgb = camera_obs['rgb']
    depth = camera_obs['depth']
    xs, zs, rgb_points, _ = get_pointcloud_from_depth(rgb, depth, camera_int)
    rgb_points_world = translate_to_world(rgb_points, camera_pos, quaternion.as_rotation_matrix(camera_rot))
    distances = np.linalg.norm(rgb_points_world - target_point, axis=1)
    closest_point_index = np.argmin(distances)
    closest_pixel_x = xs[closest_point_index]
    closest_pixel_z = zs[closest_point_index]
    closest_pixel_x_normalized = closest_pixel_x / rgb.shape[1]
    closest_pixel_z_normalized = closest_pixel_z / rgb.shape[0]
    return closest_pixel_x_normalized, closest_pixel_z_normalized

def unnormalize_goal_point(target_x_normalized, target_z_normalized, image_shape):
    return int(target_x_normalized * image_shape[1]), int(target_z_normalized * image_shape[0])

def apply_mask_to_image(image, mask, overlay_color=np.array([0, 0, 255], dtype=np.uint8)):
    result_image = image.copy()
    result_image[mask == 1] = overlay_color
    return result_image

def main(args):
    config_path = args.config_path
    if not os.path.exists(config_path):
        raise RuntimeError(f"{config_path} does not exist!")

    habitat_config = habitat.get_config(config_path)

    data_path = f"{args.data_dir}/{args.stage}/{args.stage}.json.gz"
    scene_dataset = f"{args.scene_dataset_dir}/hm3d_annotated_basis.scene_dataset_config.json"

    if not os.path.exists(data_path) or not os.path.exists(scene_dataset):
        raise RuntimeError(f"Data path or scene dataset does not exist!")

    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = args.split
        habitat_config.habitat.dataset.scenes_dir = args.scenes_dir
        habitat_config.habitat.dataset.data_path = data_path
        habitat_config.habitat.simulator.scene_dataset = scene_dataset
        habitat_config.habitat.environment.iterator_options.num_episode_sample = args.num_sampled_episodes
        habitat_config.habitat.simulator.agents.main_agent.height = args.robot_height
        habitat_config.habitat.simulator.agents.main_agent.radius = args.robot_radius
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = args.image_height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = args.image_width
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = args.image_hfov
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0, args.sensor_height, 0]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = args.image_height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = args.image_width
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = args.image_hfov
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0, args.sensor_height, 0]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = 500.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = 0.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
        habitat_config.habitat.simulator.forward_step_size = args.step_size
        habitat_config.habitat.simulator.turn_angle = args.turn_angle

    env = habitat.Env(habitat_config)
    follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)

    episode_counter = 0
    time_before_loop = time.time()

    hdf5_file_path = args.save_path

    # create parent directories
    directory_path = Path(hdf5_file_path).parent
    directory_path.mkdir(parents=True, exist_ok=True)

    # Creation loop
    with h5py.File(hdf5_file_path, 'w') as hdf:
        while episode_counter < args.num_saved_episodes:
            start_time = time.time()

            try:
                obs = env.reset()
            except Exception as e:
                print(f"An error occurred while resetting environment: {e}")
                continue

            rgb_data, depth_data, pose_data, action_data, normed_target_point_data = [], [], [], [], []
            timesteps = 0

            goal_flag, goal_image, goal_mask, goal_point, height_uncorrected_goal_point = random_pixel_goal(habitat_config, env, args.mask_shape)
            if not goal_flag:
                print(f"Rejected the current goal with goal-flag {goal_flag}")
                continue

            best_action = follower.get_next_action(goal_point)

            if best_action == 0:
                print(f"Rejected the current goal with goal-flag {goal_flag} and best-action {best_action}")
                continue

            start_rgb_image = obs["rgb"]
            start_depth_image = obs["depth"]

            last_best_action = None
            
            while True:
                assert(best_action is not None and last_best_action != 0) # sanity check

                action_data.append(best_action)
                obs = env.step(best_action)
                
                rgb_data.append(obs['rgb'])
                depth_data.append(obs['depth'])

                q = env.sim.get_agent_state().sensor_states["depth"].rotation
                pose = np.concatenate([
                    np.array(env.sim.get_agent_state().sensor_states["depth"].position),
                    np.array([q.w, q.x, q.y, q.z])
                ])

                pose_data.append(pose)
                normed_target_point_data.append(
                    get_normalized_goal_point_location_in_current_obs(habitat_config, env, height_uncorrected_goal_point)
                )

                timesteps += 1

                if not (env.episode_over or timesteps >= args.max_timesteps):
                    last_best_action = best_action
                    best_action = follower.get_next_action(goal_point)
                else:
                    print(f"Goal reached in {timesteps} steps")
                    if timesteps < args.min_timesteps:
                        print(f"{timesteps} less than min. timesteps of {args.min_timesteps}")
                        break

                    episode_group = hdf.create_group(f"episode_{episode_counter}")

                    episode_group.create_dataset("start_rgb_image", data=start_rgb_image)
                    # episode_group.create_dataset("start_depth_image", data=start_depth_image)
                    episode_group.create_dataset("goal_mask", data=goal_mask)
                    # episode_group.create_dataset("poses", data=np.array(pose_data))
                    episode_group.create_dataset("actions", data=np.array(action_data))
                    episode_group.create_dataset("normalized_target_points", data=np.array(normed_target_point_data))
                    episode_group.create_dataset("rgb_images", data=np.array(rgb_data))
                    # episode_group.create_dataset("depth_images", data=np.array(depth_data))

                    episode_counter += 1
                    end_time = time.time()
                    time_taken = end_time - start_time 
                    total_time_taken = end_time - time_before_loop
                    total_hours, rem = divmod(total_time_taken, 3600)
                    total_minutes, total_seconds = divmod(rem, 60)

                    print(f"Done with episode {episode_counter} out of {args.num_saved_episodes} total")
                    print(f"Time taken for this episode: {time_taken:.2f} seconds")
                    print(f"Total time taken: {int(total_hours)} hours, {int(total_minutes)} minutes, {int(total_seconds)} seconds")
                    print()
                    break

    print(f"All episodes saved to {hdf5_file_path}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat Data Collection Script")
    parser.add_argument("--stage", type=str, default="train", help="Stage (train/val/minival)")
    parser.add_argument("--split", type=str, default="train", help="Split (train/val/valmini)")
    parser.add_argument("--num_sampled_episodes", "-nse", type=int, default=int(1e4), help="Number of sampled episodes")
    parser.add_argument("--max_timesteps", type=int, default=64, help="Maximum timesteps per episode")
    parser.add_argument("--min_timesteps", type=int, default=5, help="Minimum timesteps per episode")
    parser.add_argument("--mask_shape", type=int, default=3, help="Shape of the goal mask")
    parser.add_argument("--config_path", type=str, default="hm3d_config_instance_image_nav_mod.yaml", help="Path to Habitat config file")
    parser.add_argument("--robot_height", type=float, default=0.88, help="Robot height")
    parser.add_argument("--robot_radius", type=float, default=0.25, help="Robot radius")
    parser.add_argument("--sensor_height", type=float, default=0.88, help="Sensor height")
    parser.add_argument("--image_width", type=int, default=224, help="Image width") # don't change
    parser.add_argument("--image_height", type=int, default=224, help="Image height")
    parser.add_argument("--image_hfov", type=float, default=79, help="Image horizontal field of view")
    parser.add_argument("--step_size", type=float, default=0.25, help="Step size")
    parser.add_argument("--turn_angle", type=float, default=30, help="Turn angle in degrees")

    parser.add_argument("--data_dir", type=str, default="/scratch/vineeth.bhat/sg_habitat/data/datasets/instance_imagenav/hm3d/v3", help="Path to data directory")
    parser.add_argument("--scene_dataset_dir", type=str, default="/scratch/vineeth.bhat/sg_habitat/data/scene_datasets/hm3d", help="Path to scene dataset directory")
    parser.add_argument("--scenes_dir", type=str, default="/scratch/vineeth.bhat/sg_habitat/data/scene_datasets", help="Path to scenes directory")
    parser.add_argument("--save_path", "-s", type=str, default="/scratch/vineeth.bhat/pix_nav_point_based_data/training_100.h5", help="Path to save data")
    parser.add_argument("--num_saved_episodes", "-n", type=int, default=int(1), help="Number of saved episodes")

    args = parser.parse_args()
    main(args)