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
from utils import *

seed_everything(42)


def augment_data(hdf5_file_path, output_dir, habitat_config):
    env = habitat.Env(habitat_config)
    env.seed(42)

    print("Initialized habitat environment")

    with h5py.File(hdf5_file_path, "r") as hdf:
        num_episodes = len(hdf.keys())

        for episode_idx in tqdm(range(num_episodes), desc="Processing episodes"):
            episode_group = hdf[f"episode_{episode_idx}"]
            episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)

            start_rgb_image = episode_group["start_rgb_image"][:]
            start_rgb_image_pil = Image.fromarray(start_rgb_image)
            start_rgb_image_path = os.path.join(episode_dir, "start_rgb_image.png")
            start_rgb_image_pil.save(start_rgb_image_path)

            start_pose = episode_group["start_pose"][:]

            print(
                f"Resetting environment for episode {episode_idx} with pose: {start_pose}"
            )

            # Reset the environment
            _ = env.reset()

            observations = env.sim.get_observations_at(
                position=start_pose[:3],
                rotation=quaternion_from_coeff(start_pose[3:]),
            )

            generated_start_rgb = observations["rgb"]
            generated_start_rgb_pil = Image.fromarray(generated_start_rgb)
            generated_start_rgb_path = os.path.join(
                episode_dir, "generated_start_rgb_image.png"
            )
            generated_start_rgb_pil.save(generated_start_rgb_path)

            rgb_images = episode_group["rgb_images"][:]
            poses = episode_group["poses"][:]

            for img_idx in range(rgb_images.shape[0]):
                rgb_image = rgb_images[img_idx]
                rgb_image_pil = Image.fromarray(rgb_image)
                rgb_image_path = os.path.join(episode_dir, f"rgb_image_{img_idx}.png")
                rgb_image_pil.save(rgb_image_path)

                pose = poses[img_idx]
                observations = env.sim.get_observations_at(
                    position=pose[:3], rotation=quaternion_from_coeff(pose[3:])
                )
                generated_rgb = observations["rgb"]
                generated_rgb_pil = Image.fromarray(generated_rgb)
                generated_rgb_path = os.path.join(
                    episode_dir, f"generated_rgb_image_{img_idx}.png"
                )
                generated_rgb_pil.save(generated_rgb_path)

            break
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

    habitat_config = habitat_config = create_habitat_config(args.config_path, args)

    augment_data(args.hdf5_file, args.output_dir, habitat_config)


if __name__ == "__main__":
    main()
