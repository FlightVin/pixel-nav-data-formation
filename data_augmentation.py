import h5py
import numpy as np
import torch
from pprint import pprint
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from tqdm import tqdm
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.config.read_write import read_write
import habitat_sim
from habitat.config.default import get_config


def get_camera_intrinsic_matrix(hfov, width, height):
    focal_length = (width / 2) / np.tan(np.deg2rad(hfov / 2))
    return np.array(
        [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]]
    )


def unproject_2d_to_3d(x, y, depth, K):
    z = depth
    x = (x - K[0, 2]) * z / K[0, 0]
    y = (y - K[1, 2]) * z / K[1, 1]
    return np.array([x, y, z])


def transform_points(points, rotation, translation):
    quaternion = quaternion_from_coeff(rotation)
    return quaternion_rotate_vector(quaternion, points) + translation


def get_pathlength_GT(sim, agent_state, depth, semantic, goal_position, display=False):
    H, W = depth.shape
    hfov = sim.config.agents[sim.default_agent_id].sensor_specifications[0].hfov
    K = get_camera_intrinsic_matrix(hfov, W, H)

    instances = np.unique(semantic)
    num_samples = 20
    area_thresh = int(np.ceil(0.001 * H * W))

    p2d_c = []
    inds, points_all = [], []
    for i, insta_idx in enumerate(instances):
        points = np.argwhere(semantic == insta_idx)
        if len(points) > num_samples:
            sub_inds = np.random.choice(len(points), num_samples, replace=False)
            points = points[sub_inds]
        p2d_c.append(points)
        inds.append(i * np.ones(len(points)))
        points_all.append(points)
    inds = np.concatenate(inds)
    p2d_c = np.concatenate(p2d_c, 0)

    p3d_c = np.array(
        [unproject_2d_to_3d(p[1], p[0], depth[p[0], p[1]], K) for p in p2d_c]
    )
    p3d_w = transform_points(p3d_c, agent_state.rotation, agent_state.position)

    p_w_nav = np.array([sim.pathfinder.snap_point(p) for p in p3d_w])
    path_follower = ShortestPathFollower(sim, goal_radius=0.05, return_one_hot=False)
    pls = []
    for p in p_w_nav:
        path = habitat.tasks.nav.shortest_path.ShortestPathPoint(p, goal_position)
        pls.append(path.geodesic_distance)

    pls = np.array(pls)

    euc_dists_agent_to_p3dw = np.linalg.norm(agent_state.position - p3d_w, axis=1)
    euc_dists_agent_to_pwnav = np.linalg.norm(agent_state.position - p_w_nav, axis=1)
    dists_mask = euc_dists_agent_to_p3dw > euc_dists_agent_to_pwnav

    pls_img = np.zeros([H, W])
    pl_min_insta = []
    for i in range(len(instances)):
        sub_inds = inds == i
        pls_insta = pls[sub_inds]
        dists_mask_insta = dists_mask[sub_inds]
        if (
            dists_mask_insta.sum() == 0
            or len(points_all[i]) <= area_thresh
            or instances[i] == 0
        ):
            pl_min = 99
        else:
            pl_min = np.min(pls_insta[dists_mask_insta])
        pl_min_insta.append(pl_min)
        pls_img[points_all[i][:, 0], points_all[i][:, 1]] = pl_min

    if display:
        plt.imshow(pls_img * (pls_img < 99))
        plt.colorbar()
        plt.show()

    return pls_img


def augment_data(hdf5_file_path, output_dir, habitat_config):
    with habitat.Env(config=habitat_config) as env:
        print("Init-ed habitat environment")
        with h5py.File(hdf5_file_path, "r") as hdf:
            num_episodes = len(hdf.keys())

            for episode_idx in tqdm(range(num_episodes), desc="Processing episodes"):
                episode_group = hdf[f"episode_{episode_idx}"]

                episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
                os.makedirs(episode_dir, exist_ok=True)

                rgb_images = episode_group["rgb_images"][()]
                normalized_target_points = episode_group["normalized_target_points"][()]

                for step, (rgb_image, norm_target) in enumerate(
                    zip(rgb_images, normalized_target_points)
                ):
                    rgb_pil = Image.fromarray(rgb_image)
                    rgb_pil.save(os.path.join(episode_dir, f"rgb_{step:03d}.png"))

                    # Reset the environment to get a valid state
                    obs = env.reset()

                    # Set the agent's state to match the current step
                    agent_state = env.sim.get_agent_state()
                    env.sim.set_agent_state(agent_state.position, agent_state.rotation)

                    # Get observations
                    obs = env.sim.get_observations_at(
                        agent_state.position, agent_state.rotation
                    )
                    depth = obs["depth"]
                    semantic = obs["semantic"]

                    # Convert normalized target to world coordinates
                    scene_bounds = env.sim.pathfinder.get_bounds()
                    goal_position = np.array(
                        [
                            norm_target[0] * (scene_bounds[1][0] - scene_bounds[0][0])
                            + scene_bounds[0][0],
                            0,  # Assume goal is on the ground
                            norm_target[1] * (scene_bounds[1][2] - scene_bounds[0][2])
                            + scene_bounds[0][2],
                        ]
                    )
                    goal_position = env.sim.pathfinder.snap_point(goal_position)

                    pls_img = get_pathlength_GT(
                        env.sim,
                        agent_state,
                        depth,
                        semantic,
                        goal_position,
                        display=False,
                    )

                    # Normalize and save mask
                    mask = (
                        (pls_img - pls_img.min())
                        / (pls_img.max() - pls_img.min())
                        * 255
                    )
                    mask_pil = Image.fromarray(mask.astype(np.uint8))
                    mask_pil.save(os.path.join(episode_dir, f"mask_{step:03d}.png"))

                    # Save depth image
                    depth_pil = Image.fromarray((depth * 255).astype(np.uint8))
                    depth_pil.save(os.path.join(episode_dir, f"depth_{step:03d}.png"))

                    # Save semantic image
                    semantic_pil = Image.fromarray(semantic.astype(np.uint8))
                    semantic_pil.save(
                        os.path.join(episode_dir, f"semantic_{step:03d}.png")
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment Habitat Pixel Navigation Dataset"
    )
    parser.add_argument(
        "--hdf5_file", type=str, required=True, help="Path to input HDF5 file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save augmented data"
    )

    parser.add_argument(
        "--stage", type=str, default="train", help="Stage (train/val/minival)"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split (train/val/valmini)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="hm3d_config_instance_image_nav_mod.yaml",
        help="Path to Habitat config file",
    )
    parser.add_argument("--robot_height", type=float, default=0.88, help="Robot height")
    parser.add_argument("--robot_radius", type=float, default=0.25, help="Robot radius")
    parser.add_argument(
        "--sensor_height", type=float, default=0.88, help="Sensor height"
    )
    parser.add_argument(
        "--image_width", type=int, default=224, help="Image width"
    )  # don't change
    parser.add_argument("--image_height", type=int, default=224, help="Image height")
    parser.add_argument(
        "--image_hfov", type=float, default=79, help="Image horizontal field of view"
    )
    parser.add_argument("--step_size", type=float, default=0.25, help="Step size")
    parser.add_argument(
        "--turn_angle", type=float, default=30, help="Turn angle in degrees"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/vineeth.bhat/sg_habitat/data/datasets/instance_imagenav/hm3d/v3",
        help="Path to data directory",
    )
    parser.add_argument(
        "--scene_dataset_dir",
        type=str,
        default="/scratch/vineeth.bhat/sg_habitat/data/scene_datasets/hm3d",
        help="Path to scene dataset directory",
    )
    parser.add_argument(
        "--scenes_dir",
        type=str,
        default="/scratch/vineeth.bhat/sg_habitat/data/scene_datasets",
        help="Path to scenes directory",
    )

    args = parser.parse_args()

    habitat_config = habitat.get_config(args.config_path)

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

    augment_data(args.hdf5_file, args.output_dir, habitat_config)
