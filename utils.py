import os
import numpy as np
import habitat
from habitat.config.read_write import read_write
import quaternion
import open3d as o3d
from habitat.config.default import get_config
import random


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def habitat_camera_intrinsic(config):
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    intrinsic_matrix = np.array([[f, 0, xc], [0, f, zc], [0, 0, 1]], np.float32)
    return intrinsic_matrix


def get_pointcloud_from_depth(rgb, depth, intrinsic):
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    filter_z, filter_x = np.where(depth > -1)
    depth_values = depth[filter_z, filter_x]
    pixel_z = (
        (depth.shape[0] - 1 - filter_z - intrinsic[1][2])
        * depth_values
        / intrinsic[1][1]
    )
    pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z, filter_x]
    point_values = np.stack([pixel_x, pixel_z, -pixel_y], axis=-1)
    return filter_x, filter_z, point_values, color_values


def translate_to_world(points, position, rotation):
    extrinsic = np.eye(4)
    extrinsic[0:3, 0:3] = rotation
    extrinsic[0:3, 3] = position
    world_points = np.matmul(
        extrinsic, np.concatenate((points, np.ones((points.shape[0], 1))), axis=-1).T
    ).T
    return np.array(world_points[:, 0:3])


def create_target_mask(target_x, target_z, mask_shape, depth_shape):
    min_z = max(target_z - mask_shape, 0)
    max_z = min(target_z + mask_shape, depth_shape[0])
    min_x = max(target_x - mask_shape, 0)
    max_x = min(target_x + mask_shape, depth_shape[1])
    target_mask = np.zeros((depth_shape[0], depth_shape[1]), np.uint8)
    target_mask[min_z:max_z, min_x:max_x] = 1
    return target_mask


def random_pixel_goal(habitat_config, habitat_env, mask_shape):
    camera_int = habitat_camera_intrinsic(habitat_config)
    robot_pos = habitat_env.sim.get_agent_state().position
    robot_rot = habitat_env.sim.get_agent_state().rotation
    camera_pos = habitat_env.sim.get_agent_state().sensor_states["rgb"].position
    camera_rot = habitat_env.sim.get_agent_state().sensor_states["rgb"].rotation
    camera_obs = habitat_env.sim.get_observations_at(robot_pos, robot_rot)
    rgb = camera_obs["rgb"]
    depth = camera_obs["depth"]
    xs, zs, rgb_points, rgb_colors = get_pointcloud_from_depth(rgb, depth, camera_int)
    rgb_points = translate_to_world(
        rgb_points, camera_pos, quaternion.as_rotation_matrix(camera_rot)
    )
    condition_index = np.where(
        (rgb_points[:, 1] < robot_pos[1] + 1.0)
        & (rgb_points[:, 1] > robot_pos[1] - 0.2)
        & (depth[(zs, xs)][:, 0] > 1.0)
        & (depth[(zs, xs)][:, 0] < 5.5)
    )[
        0
    ]  # note - using different condition indices since different task dataset

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


def get_normalized_goal_point_location_in_current_obs(
    habitat_config, habitat_env, target_point
):
    # https://github.com/wzcai99/Pixel-Navigator/issues/8#issuecomment-2378593390
    camera_int = habitat_camera_intrinsic(habitat_config)
    robot_pos = habitat_env.sim.get_agent_state().position
    robot_rot = habitat_env.sim.get_agent_state().rotation
    camera_pos = habitat_env.sim.get_agent_state().sensor_states["rgb"].position
    camera_rot = habitat_env.sim.get_agent_state().sensor_states["rgb"].rotation
    camera_obs = habitat_env.sim.get_observations_at(robot_pos, robot_rot)
    rgb = camera_obs["rgb"]
    depth = camera_obs["depth"]
    xs, zs, rgb_points, _ = get_pointcloud_from_depth(rgb, depth, camera_int)
    rgb_points_world = translate_to_world(
        rgb_points, camera_pos, quaternion.as_rotation_matrix(camera_rot)
    )
    distances = np.linalg.norm(rgb_points_world - target_point, axis=1)
    closest_point_index = np.argmin(distances)
    closest_pixel_x = xs[closest_point_index]
    closest_pixel_z = zs[closest_point_index]
    closest_pixel_x_normalized = closest_pixel_x / rgb.shape[1]
    closest_pixel_z_normalized = closest_pixel_z / rgb.shape[0]
    return closest_pixel_x_normalized, closest_pixel_z_normalized


def unnormalize_goal_point(target_x_normalized, target_z_normalized, image_shape):
    return int(target_x_normalized * image_shape[1]), int(
        target_z_normalized * image_shape[0]
    )


def apply_mask_to_image(
    image, mask, overlay_color=np.array([0, 0, 255], dtype=np.uint8)
):
    result_image = image.copy()
    result_image[mask == 1] = overlay_color
    return result_image


def create_habitat_config(config_path, args):

    habitat_config = habitat.get_config(config_path)

    data_path = f"{args.data_dir}/{args.stage}/{args.stage}.json.gz"
    scene_dataset = (
        f"{args.scene_dataset_dir}/hm3d_annotated_basis.scene_dataset_config.json"
    )

    if not os.path.exists(data_path) or not os.path.exists(scene_dataset):
        raise RuntimeError(f"Data path or scene dataset does not exist!")

    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = args.split
        habitat_config.habitat.dataset.scenes_dir = args.scenes_dir
        habitat_config.habitat.dataset.data_path = data_path
        habitat_config.habitat.simulator.scene_dataset = scene_dataset
        habitat_config.habitat.environment.iterator_options.num_episode_sample = (
            args.num_sampled_episodes
        )
        habitat_config.habitat.simulator.agents.main_agent.height = args.robot_height
        habitat_config.habitat.simulator.agents.main_agent.radius = args.robot_radius
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = (
            args.image_height
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = (
            args.image_width
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = (
            args.image_hfov
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [
            0,
            args.sensor_height,
            0,
        ]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = (
            args.image_height
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = (
            args.image_width
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = (
            args.image_hfov
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [
            0,
            args.sensor_height,
            0,
        ]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = (
            500.0
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = (
            0.0
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = (
            False
        )
        habitat_config.habitat.simulator.forward_step_size = args.step_size
        habitat_config.habitat.simulator.turn_angle = args.turn_angle
        if (
            "semantic_sensor"
            not in habitat_config.habitat.simulator.agents.main_agent.sim_sensors
        ):
            habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor = get_config(
                "habitat/simulator/sensor/semantic_sensor"
            )

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.type = (
            "HabitatSimSemanticSensor"
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = (
            args.image_height
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = (
            args.image_width
        )
        # habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = (
        #     args.image_hfov
        # )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.position = [
            0,
            args.sensor_height,
            0,
        ]
    return habitat_config
