"""
Code used to create dataset for pixel goals. Uses latest habitat lab version.
"""

import os
import argparse
import numpy as np
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import time
from pathlib import Path
import h5py
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_to_list
from utils import *

seed_everything(42)


def main(args):
    config_path = args.config_path
    if not os.path.exists(config_path):
        raise RuntimeError(f"{config_path} does not exist!")

    habitat_config = create_habitat_config(config_path, args)

    env = habitat.Env(habitat_config)
    env.seed(42)
    follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)

    episode_counter = 0
    time_before_loop = time.time()

    hdf5_file_path = args.save_path

    # create parent directories
    directory_path = Path(hdf5_file_path).parent
    directory_path.mkdir(parents=True, exist_ok=True)

    # Creation loop
    with h5py.File(hdf5_file_path, "w") as hdf:
        while episode_counter < args.num_saved_episodes:
            start_time = time.time()

            try:
                obs = env.reset()
            except Exception as e:
                print(f"An error occurred while resetting environment: {e}")
                continue

            rgb_data, depth_data, pose_data, action_data, normed_target_point_data = (
                [],
                [],
                [],
                [],
                [],
            )
            timesteps = 0

            (
                goal_flag,
                goal_image,
                goal_mask,
                goal_point,
                height_uncorrected_goal_point,
            ) = random_pixel_goal(habitat_config, env, args.mask_shape)
            if not goal_flag:
                print(f"Rejected the current goal with goal-flag {goal_flag}")
                continue

            best_action = follower.get_next_action(goal_point)

            if best_action == 0:
                print(
                    f"Rejected the current goal with goal-flag {goal_flag} and best-action {best_action}"
                )
                continue

            start_rgb_image = obs["rgb"]
            # start_semantic_image = np.squeeze(obs["semantic"])
            # print(np.unique(start_semantic_image))
            start_depth_image = obs["depth"]
            start_pose = np.concatenate(
                [
                    np.array(env.sim.get_agent_state().position),
                    np.array(quaternion_to_list(env.sim.get_agent_state().rotation)),
                ]
            )

            last_best_action = None

            while True:
                assert best_action is not None and last_best_action != 0  # sanity check

                action_data.append(best_action)
                obs = env.step(best_action)

                rgb_data.append(obs["rgb"])
                depth_data.append(obs["depth"])

                pose = np.concatenate(
                    [
                        np.array(env.sim.get_agent_state().position),
                        np.array(
                            quaternion_to_list(env.sim.get_agent_state().rotation)
                        ),
                    ]
                )

                # semantic_image = np.squeeze(obs["semantic"])
                # print(np.unique(semantic_image))

                pose_data.append(pose)
                normed_target_point_data.append(
                    get_normalized_goal_point_location_in_current_obs(
                        habitat_config, env, height_uncorrected_goal_point
                    )
                )

                # gen_observations = env.sim.get_observations_at(
                #     position=pose[:3],
                #     rotation=quaternion_from_coeff(pose[3:]),
                # )
                # print(
                #     "Images are equal",
                #     np.array_equal(obs["rgb"], gen_observations["rgb"]),
                # )

                timesteps += 1

                if not (env.episode_over or timesteps >= args.max_timesteps):
                    last_best_action = best_action
                    best_action = follower.get_next_action(goal_point)
                else:
                    print(f"Goal reached in {timesteps} steps")
                    if timesteps < args.min_timesteps:
                        print(
                            f"{timesteps} less than min. timesteps of {args.min_timesteps}"
                        )
                        break

                    episode_group = hdf.create_group(f"episode_{episode_counter}")

                    episode_group.create_dataset(
                        "start_rgb_image", data=start_rgb_image
                    )
                    # episode_group.create_dataset("start_depth_image", data=start_depth_image)
                    episode_group.create_dataset("goal_mask", data=goal_mask)
                    episode_group.create_dataset("start_pose", data=start_pose)
                    episode_group.create_dataset("poses", data=np.array(pose_data))
                    episode_group.create_dataset("actions", data=np.array(action_data))
                    episode_group.create_dataset(
                        "normalized_target_points",
                        data=np.array(normed_target_point_data),
                    )
                    episode_group.create_dataset("rgb_images", data=np.array(rgb_data))
                    episode_group.create_dataset(
                        "height_uncorrected_goal_point",
                        data=np.array(height_uncorrected_goal_point),
                    )
                    episode_group.create_dataset(
                        "goal_point",
                        data=np.array(goal_point),
                    )
                    # episode_group.create_dataset("depth_images", data=np.array(depth_data))

                    episode_counter += 1
                    end_time = time.time()
                    time_taken = end_time - start_time
                    total_time_taken = end_time - time_before_loop
                    total_hours, rem = divmod(total_time_taken, 3600)
                    total_minutes, total_seconds = divmod(rem, 60)

                    print(
                        f"Done with episode {episode_counter} out of {args.num_saved_episodes} total"
                    )
                    print(f"Time taken for this episode: {time_taken:.2f} seconds")
                    print(
                        f"Total time taken: {int(total_hours)} hours, {int(total_minutes)} minutes, {int(total_seconds)} seconds"
                    )
                    print()
                    break

    print(f"All episodes saved to {hdf5_file_path}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat Data Collection Script")
    parser.add_argument(
        "--stage", type=str, default="train", help="Stage (train/val/minival)"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split (train/val/valmini)"
    )
    parser.add_argument(
        "--num_sampled_episodes",
        "-nse",
        type=int,
        default=int(1e4),
        help="Number of sampled episodes",
    )
    parser.add_argument(
        "--max_timesteps", type=int, default=64, help="Maximum timesteps per episode"
    )
    parser.add_argument(
        "--min_timesteps", type=int, default=5, help="Minimum timesteps per episode"
    )
    parser.add_argument(
        "--mask_shape", type=int, default=3, help="Shape of the goal mask"
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
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        default="/scratch/vineeth.bhat/pix_nav_point_based_data/training_100.h5",
        help="Path to save data",
    )
    parser.add_argument(
        "--num_saved_episodes",
        "-n",
        type=int,
        default=int(1),
        help="Number of saved episodes",
    )

    args = parser.parse_args()
    main(args)
