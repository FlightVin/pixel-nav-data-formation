import sys

sys.path.insert(0, "../sg_habitat")


from dat_formation_utils import *

seed_everything()


import os
import argparse
import numpy as np
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import time
from pathlib import Path
from tqdm import tqdm
import h5py
import habitat_sim
import traceback

from joint_original_episode_goal import find_shortest_path, get_pathlength_GT_modified

from mask_goal_based_transformer_encoder.dataset import MaskEncoderDataset
from mask_goal_based_transformer_encoder.prediction_model import MaskGoalEncoder, torch


"""
/scratch/vineeth.bhat/sg_habitat/checkpoints/mask_encoder_run_10K_traj_training_Nov_19/backup_model_ckpt_0003.pt
"""


def main(args):
    config_path = args.config_path
    if not os.path.exists(config_path):
        raise RuntimeError(f"{config_path} does not exist!")

    habitat_config = create_habitat_config(config_path, args)
    env = habitat.Env(habitat_config)
    env.seed(UTILS_SEED)

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = args.robot_radius / 5
    navmesh_settings.cell_height = 0.05
    navmesh_settings.cell_size = 0.05
    navmesh_success = env.sim.recompute_navmesh(env.sim.pathfinder, navmesh_settings)

    assert navmesh_success == True

    if args.stage == "val_mini":
        ignored_objects_data = MINI_VAL_IGNORED_OBJECTS_DATA
    elif args.stage == "val":
        ignored_objects_data = VAL_IGNORED_OBJECTS_DATA
    elif args.stage == "train":
        ignored_objects_data = TRAIN_IGNORED_OBJECTS_DATA
    else:
        raise RuntimeError(f"Unknown stage {args.stage}")

    ignored_objs = [int(key.split("_")[1]) for key in ignored_objects_data.keys()]

    follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)

    episode_counter = 0
    time_before_loop = time.time()

    mask_encoder_nav_model = MaskGoalEncoder(
        max_image_history_length=16, action_output_dim=3
    )
    checkpoint = torch.load(args.checkpoint_path)
    new_state_dict = {
        k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()
    }
    mask_encoder_nav_model.load_state_dict(new_state_dict)
    mask_encoder_nav_model.eval()
    print("Loaded model!")

    pbar = tqdm(total=args.num_eval_episodes)

    def load_image(img_array):
        return torch.tensor(img_array, dtype=torch.uint8).permute(2, 0, 1)

    def load_pls(pls_array):
        return torch.tensor(pls_array, dtype=torch.uint8).unsqueeze(0)

    with torch.no_grad():
        while episode_counter < args.num_eval_episodes:
            start_time = time.time()
            print("OKKKK 1")

            try:
                start_obs = env.reset()
            except Exception as e:
                print(f"An error occurred while resetting environment: {e}")
                continue

            print("OKKKK 2")

            current_goal_obj = random.choice(env.current_episode.goals)
            unsnapped_goal_position = current_goal_obj.position

            # snap it to navigable places
            goal_position = env.sim.pathfinder.snap_point(unsnapped_goal_position)
            goal_object_id = current_goal_obj.object_id

            if goal_object_id in ignored_objs:
                print(f"{goal_object_id} in ignored objects")
                continue

            timesteps = 0

            print("OKKKK 3")

            # Create goal mask at the projected goal position
            camera_int = habitat_camera_intrinsic(habitat_config)
            camera_pos = env.sim.get_agent_state().sensor_states["rgb"].position
            camera_rot = env.sim.get_agent_state().sensor_states["rgb"].rotation

            # Project 3D goal point to 2D image coordinates
            goal_point_homogeneous = np.array(
                [goal_position[0], goal_position[1], goal_position[2], 1.0]
            )
            camera_matrix = np.eye(4)
            camera_matrix[:3, :3] = quaternion.as_rotation_matrix(camera_rot)
            camera_matrix[:3, 3] = camera_pos
            goal_camera = np.linalg.inv(camera_matrix) @ goal_point_homogeneous
            goal_camera = goal_camera[:3] / goal_camera[3]

            start_rgb = start_obs["rgb"]
            start_semantic = np.squeeze(start_obs["semantic"])
            start_depth = np.squeeze(start_obs["depth"])
            print("OKKKK 4")

            _, _, start_plsImgDirect = get_pathlength_GT_modified(
                env.sim,
                habitat_config,
                start_depth,
                start_semantic,
                goal_position,
                goal_object_id,
                ignored_objs,
            )

            start_plsImg = convert_direct_pls_image_to_uint(start_plsImgDirect)

            episode_images = load_image(start_rgb).unsqueeze(0)
            mask_images = load_pls(start_plsImg).unsqueeze(0)

            length_traj = 16

            # pad
            padding_length = length_traj - 1
            episode_images_padding = torch.zeros(
                padding_length,
                *episode_images.shape[1:],
                dtype=episode_images.dtype,
            )
            episode_images = torch.cat([episode_images, episode_images_padding], dim=0)
            pls_images_padding = torch.zeros(
                padding_length,
                *mask_images.shape[1:],
                dtype=mask_images.dtype,
            )
            mask_images = torch.cat([mask_images, pls_images_padding], dim=0)

            # Batch
            print(episode_images.shape, mask_images.shape)

            episode_images = episode_images[None, :]
            mask_images = mask_images[None, :]

            episode_images.to(mask_encoder_nav_model.device)
            mask_images.to(mask_encoder_nav_model.device)
            print(episode_images.shape, mask_images.shape)

            cur_action_index = 0

            cur_action = (
                int(
                    mask_encoder_nav_model(episode_images, mask_images)[0][
                        cur_action_index
                    ]
                    .argmax(dim=-1)
                    .cpu()
                )
                + 1
            )

            while True:
                print(
                    f"Taking {cur_action} action at timestep {timesteps} and action index {cur_action_index}"
                )
                current_obs = env.step(cur_action)

                cur_image = current_obs["rgb"]
                cur_sem = np.squeeze(current_obs["semantic"])
                cur_depth = np.squeeze(current_obs["depth"])

                _, _, cur_plsImgDirect = get_pathlength_GT_modified(
                    env.sim,
                    habitat_config,
                    cur_depth,
                    cur_sem,
                    goal_position,
                    goal_object_id,
                    ignored_objs,
                )

                cur_plsImg = convert_direct_pls_image_to_uint(cur_plsImgDirect)

                # cur_action_index += 1
                # cur_action_index = min(cur_action_index, length_traj - 1)

                if cur_action_index == length_traj - 1:
                    episode_padding = torch.zeros(
                        1,
                        *episode_images[0].shape[1:],
                        dtype=episode_images.dtype,
                    )
                    episode_images[0] = torch.cat(
                        episode_images[0][1:], episode_padding[0], dim=0
                    )

                    pls_padding = torch.zeros(
                        1,
                        *mask_images[0].shape[1:],
                        dtype=mask_images.dtype,
                    )
                    mask_images[0] = torch.cat(
                        mask_images[0][1:], pls_padding[0], dim=0
                    )
                else:
                    cur_action_index += 1

                cur_rgb_tensor = load_image(cur_image).to(mask_encoder_nav_model.device)
                cur_pls_tensor = load_pls(cur_plsImg).to(mask_encoder_nav_model.device)

                episode_images[0][cur_action_index] = cur_rgb_tensor
                mask_images[0][cur_action_index] = cur_pls_tensor

                cur_action = (
                    int(
                        mask_encoder_nav_model(episode_images, mask_images)[0][
                            cur_action_index
                        ]
                        .argmax(dim=-1)
                        .cpu()
                    )
                    + 1
                )

                timesteps += 1

            episode_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat Data Collection Script")
    parser.add_argument(
        "--stage", type=str, default="train", help="Stage (train/val/minival)"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split (train/val/valmini)"
    )
    parser.add_argument("--num_sampled_episodes", "-nse", type=int, default=10)
    parser.add_argument("--max_timesteps", type=int, default=96)
    parser.add_argument("--mask_shape", type=int, default=3)
    parser.add_argument(
        "--config_path", type=str, default="hm3d_config_instance_image_nav_mod.yaml"
    )
    parser.add_argument("--robot_height", type=float, default=0.88)
    parser.add_argument("--robot_radius", type=float, default=0.30)
    parser.add_argument("--sensor_height", type=float, default=0.88)
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=120)
    parser.add_argument("--image_hfov", type=float, default=79)
    parser.add_argument("--step_size", type=float, default=0.15)
    parser.add_argument("--turn_angle", type=float, default=30)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/vineeth.bhat/sg_habitat/data/datasets/instance_imagenav/hm3d/v3",
    )
    parser.add_argument(
        "--scene_dataset_dir",
        type=str,
        default="/scratch/vineeth.bhat/sg_habitat/data/scene_datasets/hm3d",
    )
    parser.add_argument(
        "--scenes_dir",
        type=str,
        default="/scratch/vineeth.bhat/sg_habitat/data/scene_datasets",
    )

    # For the validation run
    parser.add_argument("--num_eval_episodes", "-n", type=int, default=int(1))
    parser.add_argument("--checkpoint_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
