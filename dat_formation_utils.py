import os
import numpy as np
import habitat
from habitat.config.read_write import read_write
import quaternion
import open3d as o3d
from habitat.config.default import get_config
import random
from PIL import Image

UTILS_SEED = 42
SEMANTIC_COLOR_MAP = {}


def seed_everything(seed=UTILS_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def habitat_camera_intrinsic(config):
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    fy = (height / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    intrinsic_matrix = np.array([[fx, 0, xc], [0, fy, zc], [0, 0, 1]], np.float32)
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
        & (depth[(zs, xs)][:, 0] > 0.5)
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


def create_habitat_config(config_path, args, seed=UTILS_SEED):

    habitat_config = habitat.get_config(config_path, overrides=[f"habitat.seed={seed}"])

    data_path = f"{args.data_dir}/{args.stage}/{args.stage}.json.gz"
    scene_dataset = (
        f"{args.scene_dataset_dir}/hm3d_annotated_basis.scene_dataset_config.json"
    )
    print(f"Using data path: {data_path}")
    print(f"Using scene dataset: {scene_dataset}")

    if not os.path.exists(data_path):
        raise RuntimeError(f"Data path path does not exist: {data_path}")
    if not os.path.exists(scene_dataset):
        raise RuntimeError(f"Scene dataset path does not exist: {scene_dataset}")

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
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = (
            args.image_height
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = (
            args.image_width
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.position = [
            0,
            args.sensor_height,
            0,
        ]
    return habitat_config


def create_episode_directory(base_dir, episode_idx):
    episode_dir = os.path.join(base_dir, f"episode_{episode_idx}")
    os.makedirs(episode_dir, exist_ok=True)
    return episode_dir


def save_rgb_image(image_array, save_path):
    image = Image.fromarray(image_array.astype(np.uint8))
    if save_path is not None:
        image.save(save_path)
    return image


def save_semantic_image(semantic_image, save_path):
    unique_labels = np.unique(semantic_image)

    for label in unique_labels:
        if label not in SEMANTIC_COLOR_MAP:
            SEMANTIC_COLOR_MAP[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

    height, width = semantic_image.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for label in unique_labels:
        rgb_image[semantic_image == label] = SEMANTIC_COLOR_MAP[label]

    return save_rgb_image(rgb_image, save_path)


def save_pathlength_image(image_array, save_path):
    rgb_array = np.stack((image_array,) * 3, axis=-1)
    return save_rgb_image(rgb_array, save_path)


def convert_direct_pls_image_to_uint(image_array):
    image_array = np.asarray(image_array)
    processed_array = image_array * 20
    processed_array = np.clip(processed_array, 0, 255)
    processed_array[np.isinf(processed_array)] = 255
    return processed_array.astype(np.uint8)


def save_depth_image(depth_array, save_path, max_depth=20.0):
    clipped_depth = np.clip(depth_array, 0, max_depth)
    normalized_depth = 1.0 - (clipped_depth / max_depth)
    depth_image = (normalized_depth * 255).astype(np.uint8)
    depth_image[depth_array > max_depth] = 255

    return save_rgb_image(depth_image, save_path)


def save_multiple_images_as_row(images, save_path):
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    new_image = Image.new("RGB", (total_width, max_height))

    current_x = 0
    for image in images:
        new_image.paste(image, (current_x, 0))
        current_x += image.width

    if save_path is not None:
        new_image.save(save_path)

    return new_image


MINI_VAL_IGNORED_OBJECTS_DATA = {
    "ceiling_1": "ceiling",
    "ceiling_239": "ceiling",
    "ceiling_296": "ceiling",
    "ceiling_374": "ceiling",
    "ceiling_380": "ceiling",
    "ceiling_419": "ceiling",
    "ceiling_422": "ceiling",
    "ceiling_443": "ceiling",
    "ceiling_462": "ceiling",
    "ceiling_540": "ceiling",
    "ceiling_55": "ceiling",
    "ceiling_584": "ceiling",
    "ceiling_649": "ceiling",
    "ceiling_65": "ceiling",
    "floor mat_271": "floor mat",
    "floor mat_293": "floor mat",
    "floor mat_294": "floor mat",
    "floor_134": "floor",
    "floor_289": "floor",
    "floor_292": "floor",
    "floor_31": "floor",
    "floor_362": "floor",
    "floor_363": "floor",
    "floor_375": "floor",
    "floor_40": "floor",
    "floor_408": "floor",
    "floor_444": "floor",
    "floor_532": "floor",
    "floor_549": "floor",
    "floor_591": "floor",
    "floor_644": "floor",
}

VAL_IGNORED_OBJECTS_DATA = {
    "floor_22": "floor",
    "floor_267": "floor",
    "floor_290": "floor",
    "floor_318": "floor",
    "floor_367": "floor",
    "floor_402": "floor",
    "floor_412": "floor",
    "floor_426": "floor",
    "floor_490": "floor",
    "floor_510": "floor",
    "floor_536": "floor",
    "floor_69": "floor",
    "floor_97": "floor",
    "ceiling_1": "ceiling",
    "ceiling_170": "ceiling",
    "ceiling_227": "ceiling",
    "ceiling_263": "ceiling",
    "ceiling_288": "ceiling",
    "ceiling_307": "ceiling",
    "ceiling_361": "ceiling",
    "ceiling_37": "ceiling",
    "ceiling_38": "ceiling",
    "ceiling_389": "ceiling",
    "ceiling_414": "ceiling",
    "ceiling_461": "ceiling",
    "ceiling_463": "ceiling",
    "ceiling_496": "ceiling",
    "ceiling_497": "ceiling",
    "ceiling_503": "ceiling",
    "ceiling_506": "ceiling",
    "ceiling_511": "ceiling",
    "ceiling_537": "ceiling",
}

TRAIN_IGNORED_OBJECTS_DATA = {
    "ceiling_1": "ceiling",
    "ceiling_101": "ceiling",
    "ceiling_102": "ceiling",
    "ceiling_14": "ceiling",
    "ceiling_170": "ceiling",
    "ceiling_192": "ceiling",
    "ceiling_204": "ceiling",
    "ceiling_218": "ceiling",
    "ceiling_231": "ceiling",
    "ceiling_240": "ceiling",
    "ceiling_247": "ceiling",
    "ceiling_277": "ceiling",
    "ceiling_289": "ceiling",
    "ceiling_29": "ceiling",
    "ceiling_311": "ceiling",
    "ceiling_359": "ceiling",
    "ceiling_372": "ceiling",
    "ceiling_39": "ceiling",
    "ceiling_451": "ceiling",
    "ceiling_483": "ceiling",
    "ceiling_49": "ceiling",
    "ceiling_498": "ceiling",
    "ceiling_513": "ceiling",
    "ceiling_545": "ceiling",
    "ceiling_589": "ceiling",
    "ceiling_601": "ceiling",
    "ceiling_62": "ceiling",
    "ceiling_620": "ceiling",
    "ceiling_655": "ceiling",
    "ceiling_667": "ceiling",
    "ceiling_688": "ceiling",
    "ceiling_69": "ceiling",
    "ceiling_700": "ceiling",
    "ceiling_74": "ceiling",
    "ceiling_8": "ceiling",
    "ceiling_88": "ceiling",
    "floor_103": "floor",
    "floor_171": "floor",
    "floor_193": "floor",
    "floor_203": "floor",
    "floor_217": "floor",
    "floor_230": "floor",
    "floor_248": "floor",
    "floor_278": "floor",
    "floor_297": "floor",
    "floor_312": "floor",
    "floor_34": "floor",
    "floor_360": "floor",
    "floor_373": "floor",
    "floor_40": "floor",
    "floor_452": "floor",
    "floor_484": "floor",
    "floor_50": "floor",
    "floor_500": "floor",
    "floor_514": "floor",
    "floor_546": "floor",
    "floor_590": "floor",
    "floor_602": "floor",
    "floor_621": "floor",
    "floor_63": "floor",
    "floor_656": "floor",
    "floor_689": "floor",
    "floor_75": "floor",
    "floor_9": "floor",
    "floor_91": "floor",
}
