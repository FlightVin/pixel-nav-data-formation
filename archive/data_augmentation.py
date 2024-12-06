from dat_formation_utils import *

seed_everything(UTILS_SEED)

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
from dat_formation_utils import *
import magnum as mn
from spatialmath import SE3
from spatialmath.base import trnorm


def create_episode_directories(base_dir, episode_idx):
    episode_dir = os.path.join(base_dir, f"episode_{episode_idx}")
    rgb_dir = os.path.join(episode_dir, "rgb_images")
    pls_dir = os.path.join(episode_dir, "pls_images")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(pls_dir, exist_ok=True)

    return episode_dir, rgb_dir, pls_dir


def save_rgb_image(image_array, save_path, is_rgb=True):
    if is_rgb:
        image = Image.fromarray(image_array.astype(np.uint8))
    else:
        clipped = np.clip(image_array, 0, 100)

        upper_bound = max(
            (min(np.max(clipped[clipped < 80]), 80) if np.any(clipped < 80) else 80), 5
        )
        rgb_array = np.zeros((*clipped.shape, 3))

        mask_under_upper_bound = clipped <= upper_bound
        values_under_upper_bound = clipped[mask_under_upper_bound]

        norm_under_upper_bound = (
            values_under_upper_bound / upper_bound
        ) ** 2  # Squaring for contrast enhancement

        rgb_array[mask_under_upper_bound, 1] = 1 - norm_under_upper_bound
        rgb_array[mask_under_upper_bound, 2] = norm_under_upper_bound

        rgb_array[clipped > upper_bound, 0] = 1.0

        image = Image.fromarray((rgb_array * 255).astype(np.uint8))

    image.save(save_path)


def find_shortest_path(sim, p1, p2):
    path = habitat_sim.ShortestPath()
    path.requested_start = p1
    path.requested_end = p2
    found_path = sim.pathfinder.find_path(path)
    return path.geodesic_distance, path.points


def get_pathlength_GT_modified(sim, habitat_config, depth, semantic, goal_position):
    H, W = depth.shape
    K = habitat_camera_intrinsic(habitat_config)
    instances = np.unique(semantic)
    numSamples = 50
    areaThresh = int(np.ceil(0.0005 * H * W))

    position = sim.get_agent_state().position
    rotation = sim.get_agent_state().rotation
    camera_pos = sim.get_agent_state().sensor_states["rgb"].position
    camera_rot = sim.get_agent_state().sensor_states["rgb"].rotation

    p2d_c = []
    inds, pointsAll = [], []
    for i, insta_idx in enumerate(instances):
        points = np.argwhere(semantic == insta_idx)
        if len(points) > 0:
            actual_samples = min(numSamples, len(points))
            subInds = np.linspace(0, len(points) - 1, actual_samples).astype(int)
            p2d_c.append(points[subInds])
            inds.append(i * np.ones(actual_samples))
            pointsAll.append(points)
        else:
            pointsAll.append(points)

    if not p2d_c:
        return np.array([]), {}, np.zeros([H, W])

    inds = np.concatenate(inds)
    p2d_c = np.concatenate(p2d_c, 0)

    xs = p2d_c[:, 1]  # Column coordinates
    zs = p2d_c[:, 0]  # Row coordinates
    depths = depth[zs, xs]

    xc = (W - 1.0) / 2.0
    zc = (H - 1.0) / 2.0
    fx = K[0, 0]
    fy = K[1, 1]

    p3d_c = np.zeros((len(xs), 3))
    p3d_c[:, 0] = (xs - xc) * depths / fx  # X coordinates
    p3d_c[:, 1] = (zs - zc) * depths / fy  # Y coordinates
    p3d_c[:, 2] = depths  # Z coordinates

    p3d_w = translate_to_world(
        p3d_c, camera_pos, quaternion.as_rotation_matrix(camera_rot)
    )

    p_w_nav = np.array([sim.pathfinder.snap_point(p) for p in p3d_w[:, :3]])
    pls = np.array([find_shortest_path(sim, p, goal_position)[0] for p in p_w_nav])

    eucDists_agent_to_p3dw = np.linalg.norm(position - p3d_w[:, :3], axis=1)
    eucDists_agent_to_pwnav = np.linalg.norm(position - p_w_nav, axis=1)
    distsMask = eucDists_agent_to_p3dw > eucDists_agent_to_pwnav

    plsImg = np.zeros([H, W])
    pl_min_insta = []
    for i in range(len(instances)):
        subInds = inds == i
        pls_insta = pls[subInds]
        distsMask_insta = distsMask[subInds]

        if distsMask_insta.sum() == 0:
            pl_min = np.inf
        else:
            pl_min = np.min(pls_insta[distsMask_insta])

        if pl_min == np.inf or len(pointsAll[i]) <= areaThresh or instances[i] == 0:
            pl_min = 99

        pl_min_insta.append(pl_min)
        plsImg[pointsAll[i][:, 0], pointsAll[i][:, 1]] = pl_min

    pls = np.array(pl_min_insta)
    plsDict = {instances[i]: pls[i] for i in range(len(instances))}
    plsImg = plsImg.reshape([H, W])

    return pls, plsDict, plsImg


def augment_data(hdf5_file_path, output_dir, habitat_config):
    env = habitat.Env(habitat_config)
    env.seed(UTILS_SEED)

    print("Initialized habitat environment")

    with h5py.File(hdf5_file_path, "r") as hdf:
        num_episodes = len(hdf.keys())

        for episode_idx in tqdm(
            range(num_episodes), desc="### Processing episodes ###"
        ):
            episode_group = hdf[f"episode_{episode_idx}"]
            episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)

            start_rgb_image = episode_group["start_rgb_image"][:]
            start_pose = episode_group["start_pose"][:]
            goal_position = (
                episode_group["height_uncorrected_goal_point"][:]
                if "height_uncorrected_goal_point" in episode_group
                else None
            )
            assert goal_position is not None

            print(
                f"Resetting environment for episode {episode_idx} with pose: {start_pose}"
                f" and goal position {goal_position}"
            )
            # _ = env.reset()

            observations = env.sim.get_observations_at(
                position=start_pose[:3], rotation=quaternion_from_coeff(start_pose[3:])
            )

            start_semantic_image = np.squeeze(observations["semantic"])
            start_depth_image = np.squeeze(observations["depth"])

            start_pls, start_plsDict, start_plsImg = get_pathlength_GT_modified(
                env.sim,
                habitat_config,
                start_depth_image,
                start_semantic_image,
                goal_position,
            )

            # np.save(os.path.join(episode_dir, "start_pathlengths.npy"), start_pls)
            # np.save(
            #     os.path.join(episode_dir, "start_pathlengths_image.npy"),
            #     start_plsImg,
            # )
            save_rgb_image(
                start_plsImg,
                os.path.join(episode_dir, "start_pathlengths_image.png"),
                is_rgb=False,
            )
            save_rgb_image(
                start_rgb_image, os.path.join(episode_dir, f"start_rgb_image.png")
            )
            save_rgb_image(
                observations["rgb"],
                os.path.join(episode_dir, f"start_generated_rgb_image.png"),
            )

            # print(f"Start Image")
            # print("Shape of pls", start_pls.shape)
            # print("Unique from pls", np.unique(start_pls))
            # print("Shape of plsImage", start_plsImg.shape)
            # print("Unique from plsImage", np.unique(start_plsImg))

            rgb_images = episode_group["rgb_images"][:]
            poses = episode_group["poses"][:]

            if goal_position is not None:
                trajectory_pls = []
                trajectory_plsImgs = []

            for img_idx in range(rgb_images.shape[0]):
                pose = poses[img_idx]
                observations = env.sim.get_observations_at(
                    position=pose[:3], rotation=quaternion_from_coeff(pose[3:])
                )

                semantic_image = np.squeeze(observations["semantic"])
                depth_image = np.squeeze(observations["depth"])

                pls, plsDict, plsImg = get_pathlength_GT_modified(
                    env.sim,
                    habitat_config,
                    depth_image,
                    semantic_image,
                    goal_position,
                )

                save_rgb_image(
                    plsImg,
                    os.path.join(episode_dir, f"pathlengths_image_{img_idx:05}.png"),
                    is_rgb=False,
                )
                save_rgb_image(
                    rgb_images[img_idx],
                    os.path.join(episode_dir, f"rgb_image_{img_idx:05}.png"),
                )
                save_rgb_image(
                    observations["rgb"],
                    os.path.join(episode_dir, f"generated_rgb_image_{img_idx:05}.png"),
                )

                # print(f"Episode {img_idx}")
                # print("Shape", pls.shape)
                # print("Unique from pls", np.unique(pls))
                # print("Shape of plsImage", plsImg.shape)
                # print("Unique from plsImage", np.unique(plsImg))
                # print()
                # print()

    print(f"Dataset augmented and converted. Output saved to {output_dir}")
    env.close()


def main():
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
        "--num_sampled_episodes",
        "-nse",
        type=int,
        default=int(1e4),
        help="Number of sampled episodes",
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

    habitat_config = create_habitat_config(args.config_path, args)

    augment_data(args.hdf5_file, args.output_dir, habitat_config)


if __name__ == "__main__":
    main()
