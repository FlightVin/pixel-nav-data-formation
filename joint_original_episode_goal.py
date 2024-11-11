from utils import *

seed_everything()


import os
import argparse
import numpy as np
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import time
from pathlib import Path
from tqdm import tqdm
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_to_list
import habitat_sim


# def find_shortest_path(sim, p1, p2):
#     # path = habitat_sim.ShortestPath()
#     # path.requested_start = p1
#     # path.requested_end = p2
#     # found_path = sim.pathfinder.find_path(path)

#     # # print(sim.agents[0])
#     # # raise

#     # ggf = habitat_sim.nav.GreedyGeodesicFollower(sim.pathfinder, sim.agents[0])
#     # return len(ggf.find_path(p2)), None

#     # return path.geodesic_distance, path.points

#     path = habitat_sim.ShortestPath()
#     path.requested_start = p1
#     path.requested_end = p2
#     found_path = sim.pathfinder.find_path(path)
#     return path.geodesic_distance, path.points
#     # geod_distance = sim.geodesic_distance(p1, p2)
#     # return geod_distance, None
#     # if geod_distance >= 4:
#     #     return geod_distance, None
#     # print(np.linalg.norm(np.array(p1) - np.array(p2)))
#     # print(geod_distance)
#     return min(geod_distance, np.linalg.norm(np.array(p1) - np.array(p2))), None


def find_shortest_path(sim, p1, p2, threshold=4):
    path = habitat_sim.ShortestPath()
    path.requested_start = p1
    path.requested_end = p2
    _ = sim.pathfinder.find_path(path)

    geodesic_distance = path.geodesic_distance
    if geodesic_distance >= threshold:
        return geodesic_distance, path.points

    euclidean_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    return (geodesic_distance + euclidean_distance) / 2, path.points


def get_pathlength_GT_modified(
    sim, habitat_config, depth, semantic, goal_position, goal_object_id, ignored_objs=[]
):
    H, W = depth.shape
    K = habitat_camera_intrinsic(habitat_config)
    instances = np.unique(semantic)
    numSamples = 70
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

    xs = p2d_c[:, 1]
    zs = p2d_c[:, 0]
    depths = depth[zs, xs]

    xc = (W - 1.0) / 2.0
    zc = (H - 1.0) / 2.0
    fx = K[0, 0]
    fy = K[1, 1]

    p3d_c = np.zeros((len(xs), 3))
    p3d_c[:, 0] = (xs - xc) * depths / fx
    p3d_c[:, 1] = (zs - zc) * depths / fy
    p3d_c[:, 2] = depths

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

    def heuristic_minimum(pls_insta, distsMask_insta, num_values=10):
        masked_values = np.sort(pls_insta[distsMask_insta])
        filtered_values = masked_values[np.isfinite(masked_values)]
        if len(filtered_values) == 0:
            return np.inf
        first_n_values = masked_values[:num_values]
        weights = np.exp(-np.arange(len(first_n_values)))
        weighted_average = np.average(first_n_values, weights=weights)
        return weighted_average

    for i in range(len(instances)):
        subInds = inds == i
        pls_insta = pls[subInds]
        distsMask_insta = distsMask[subInds]

        if distsMask_insta.sum() == 0 or instances[i] in ignored_objs:
            pl_min = np.inf
        else:
            # pl_min = np.min(pls_insta[distsMask_insta])
            pl_min = heuristic_minimum(pls_insta, distsMask_insta)
            if instances[i] == goal_object_id:
                pl_min = 0.75

        if (
            pl_min == np.inf
            or (len(pointsAll[i]) <= areaThresh and instances[i] != goal_object_id)
            or instances[i] == 0
        ):
            pl_min = 99

        pl_min_insta.append(pl_min)
        plsImg[pointsAll[i][:, 0], pointsAll[i][:, 1]] = pl_min

    pls = np.array(pl_min_insta)
    plsDict = {instances[i]: pls[i] for i in range(len(instances))}
    plsImg = plsImg.reshape([H, W])

    return pls, plsDict, plsImg


def main(args):
    config_path = args.config_path
    if not os.path.exists(config_path):
        raise RuntimeError(f"{config_path} does not exist!")

    habitat_config = create_habitat_config(config_path, args)
    env = habitat.Env(habitat_config)
    env.seed(UTILS_SEED)

    # semantic_scene = env.sim.semantic_scene
    # instance_id_to_name = {}
    # for obj in semantic_scene.objects:
    #     if obj is not None:
    #         instance_id_to_name[obj.id] = obj.category.name()
    # print("Semantic Instance Mapping:")
    # from pprint import pprint

    # pprint(instance_id_to_name)
    # raise

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = args.robot_radius / 5
    navmesh_settings.cell_height = 0.05
    navmesh_settings.cell_size = 0.05
    navmesh_success = env.sim.recompute_navmesh(env.sim.pathfinder, navmesh_settings)

    assert navmesh_success == True

    ignored_objs_data = {
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
    ignored_objs = [int(key.split("_")[1]) for key in ignored_objs_data.keys()]

    follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)

    episode_counter = 0
    time_before_loop = time.time()

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=args.num_saved_episodes)

    while episode_counter < args.num_saved_episodes:
        start_time = time.time()

        try:
            start_obs = env.reset()
        except Exception as e:
            print(f"An error occurred while resetting environment: {e}")
            continue
        # Get episode goal
        unsnapped_goal_position = env.current_episode.goals[0].position

        # from pprint import pprint

        # pprint(f"We have {len(env.current_episode.goals)} goals")
        # pprint("The first goal is")
        # pprint(env.current_episode.goals[0])
        # raise

        # snap it to navigable places
        goal_position = env.sim.pathfinder.snap_point(unsnapped_goal_position)
        goal_object_id = env.current_episode.goals[0].object_id

        if goal_object_id in ignored_objs:
            continue

        episode_dir = create_episode_directory(output_dir, episode_counter)
        timesteps = 0

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

        # if goal_camera[2] <= 0:  # Goal is behind camera
        #     print("Goal is behind camera, skipping episode")
        #     continue

        goal_pixel = camera_int @ goal_camera[:3]
        goal_pixel = goal_pixel / goal_pixel[2]
        goal_x, goal_z = int(goal_pixel[0]), int(goal_pixel[1])

        # Check if goal is in frame
        # if (
        #     goal_x < 0
        #     or goal_x >= start_obs["rgb"].shape[1]
        #     or goal_z < 0
        #     or goal_z >= start_obs["rgb"].shape[0]
        # ):
        #     print("Goal not in camera frame, skipping episode")
        #     continue

        goal_mask = create_target_mask(
            goal_x, goal_z, args.mask_shape, start_obs["depth"].shape
        )

        best_action = follower.get_next_action(goal_position)
        if best_action == 0:
            print(f"Rejected the current goal with best-action {best_action}")
            continue

        start_rgb = start_obs["rgb"]
        start_semantic = np.squeeze(start_obs["semantic"])
        start_depth = np.squeeze(start_obs["depth"])

        last_best_action = None

        rgb_data, depth_data, semantic_data, normed_goal_point_position_data = (
            [],
            [],
            [],
            [],
        )

        while True:
            assert best_action is not None and last_best_action != 0

            current_obs = env.step(best_action)

            # Accessing current observations
            rgb_data.append(current_obs["rgb"])
            semantic_data.append(np.squeeze(current_obs["semantic"]))
            depth_data.append(np.squeeze(current_obs["depth"]))

            # Calculate normalized goal position in current frame
            current_camera_pos = env.sim.get_agent_state().sensor_states["rgb"].position
            current_camera_rot = env.sim.get_agent_state().sensor_states["rgb"].rotation

            # Project goal into current frame
            goal_point_norm_x, goal_point_norm_z = (
                get_normalized_goal_point_location_in_current_obs(
                    habitat_config, env, goal_position
                )
            )
            normed_goal_point_position_data.append(
                [goal_point_norm_x, goal_point_norm_z]
            )

            timesteps += 1

            if not (env.episode_over or timesteps >= args.max_timesteps):
                last_best_action = best_action
                best_action = follower.get_next_action(goal_position)
            else:
                print(f"Goal reached in {timesteps} steps")
                if timesteps < args.min_timesteps:
                    print(
                        f"{timesteps} less than min. timesteps of {args.min_timesteps}"
                    )
                    break

                # Save initial data
                save_rgb_image(
                    apply_mask_to_image(start_rgb, goal_mask),
                    os.path.join(episode_dir, "start_rgb_image_with_mask.png"),
                )
                _, _, start_plsImg = get_pathlength_GT_modified(
                    env.sim,
                    habitat_config,
                    start_depth,
                    start_semantic,
                    goal_position,
                    goal_object_id,
                    ignored_objs,
                )
                save_pathlength_image(
                    start_plsImg,
                    os.path.join(episode_dir, "start_pls_image.png"),
                )

                # Save trajectory data
                for timestep_idx, (
                    current_rgb,
                    current_depth,
                    current_semantic,
                    current_normed_goal_pt_loc,
                ) in enumerate(
                    zip(
                        rgb_data,
                        depth_data,
                        semantic_data,
                        normed_goal_point_position_data,
                    )
                ):
                    current_goal_pt_loc = unnormalize_goal_point(
                        current_normed_goal_pt_loc[0],
                        current_normed_goal_pt_loc[1],
                        current_rgb.shape,
                    )
                    current_goal_point_mask = create_target_mask(
                        current_goal_pt_loc[0],
                        current_goal_pt_loc[1],
                        args.mask_shape,
                        current_depth.shape,
                    )
                    current_rgb_with_goal_mask = apply_mask_to_image(
                        current_rgb, current_goal_point_mask
                    )
                    rgb_image_pil_obj = save_rgb_image(
                        current_rgb_with_goal_mask,
                        None,
                    )

                    semantic_image_pil_obj = save_semantic_image(
                        current_semantic,
                        None,
                    )

                    _, _, current_plsImg = get_pathlength_GT_modified(
                        env.sim,
                        habitat_config,
                        current_depth,
                        current_semantic,
                        goal_position,
                        goal_object_id,
                        ignored_objs,
                    )
                    # print(np.unique(current_plsImg))

                    pathlength_image_pil_obj = save_pathlength_image(
                        current_plsImg,
                        None,
                    )

                    save_multiple_images_as_row(
                        [
                            rgb_image_pil_obj,
                            semantic_image_pil_obj,
                            pathlength_image_pil_obj,
                        ],
                        os.path.join(
                            episode_dir, f"observation_{timestep_idx:05d}.png"
                        ),
                    )

                episode_counter += 1
                pbar.update(1)

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

    print(f"All episodes saved to {output_dir}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat Data Collection Script")
    parser.add_argument(
        "--stage", type=str, default="train", help="Stage (train/val/minival)"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split (train/val/valmini)"
    )
    parser.add_argument("--num_sampled_episodes", "-nse", type=int, default=10)
    parser.add_argument("--max_timesteps", type=int, default=128)
    parser.add_argument("--min_timesteps", type=int, default=5)
    parser.add_argument("--mask_shape", type=int, default=3)
    parser.add_argument(
        "--config_path", type=str, default="hm3d_config_instance_image_nav_mod.yaml"
    )
    parser.add_argument("--robot_height", type=float, default=0.88)
    parser.add_argument("--robot_radius", type=float, default=0.30)
    parser.add_argument("--sensor_height", type=float, default=0.88)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=224)
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
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the dataset"
    )
    parser.add_argument("--num_saved_episodes", "-n", type=int, default=int(1))

    args = parser.parse_args()
    main(args)
