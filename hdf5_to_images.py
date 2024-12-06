import os
import h5py
from PIL import Image
import numpy as np
from pathlib import Path
from dat_formation_utils import *


def main(hdf5_file_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_file_path, "r") as hdf:
        for episode_idx, episode_group in enumerate(hdf.values()):
            episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)

            start_rgb_image = episode_group["start_rgb_image"][:]
            start_pls_image = episode_group["start_pls_image"][:]
            start_rgb_image_pil_obj = save_rgb_image(start_rgb_image, None)
            start_pls_image_pil_obj = save_pathlength_image(start_pls_image, None)

            save_multiple_images_as_row(
                [
                    start_rgb_image_pil_obj,
                    start_pls_image_pil_obj,
                ],
                os.path.join(episode_dir, "start.png"),
            )

            # Save trajectory data
            rgb_images = episode_group["rgb_images"][:]
            pls_images = episode_group["pls_images"][:]
            for timestep_idx, (rgb_image, pls_image) in enumerate(
                zip(rgb_images, pls_images)
            ):
                cur_rgb_image_pil_obj = save_rgb_image(rgb_image, None)
                cur_pls_image_pil_obj = save_pathlength_image(pls_image, None)

                save_multiple_images_as_row(
                    [
                        cur_rgb_image_pil_obj,
                        cur_pls_image_pil_obj,
                    ],
                    os.path.join(episode_dir, f"observation_{timestep_idx:05d}.png"),
                )

    print(f"Data saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and save data from HDF5 file")
    parser.add_argument(
        "--hdf5-file-path", type=str, required=True, help="Path to the HDF5 file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save the images",
    )
    args = parser.parse_args()

    main(args.hdf5_file_path, args.output_dir)
