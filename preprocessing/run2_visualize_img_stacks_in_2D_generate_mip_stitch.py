# %% [markdown]
# Author(s): Piyush Amitabh
#
# Details: this code generates mip and stitches them to visualize the img stacks
#
# Created: May 02, 2023
#
# License: GNU GPL v3.0

# %% [markdown]
# Comment: it is scope agnostic (KLA/WIL LSM) and os agnostic

# %% [markdown]
# Updated: 29 Sep, 23
#
# Detail: now this works with multi folder/multi acquisition

import os

import batchprocessing_helper_functions as bpf
import skimage
import tifffile as tiff
from natsort import natsorted
from tqdm import tqdm

action_flag = 0

while action_flag == 0:
    action_flag = int(
        input(
            """Do you want to:
                        1. Find Max Intensity Projection AND Stitch (default)
                        2. Only find Max Intensity Projection
                        3. Only Stitch\n"""
        )
        or "1"
    )
    if action_flag == 1 or action_flag == 2 or action_flag == 3:
        break
    else:
        action_flag = 0
        print("Invalid value: Re-Enter")

if action_flag != 2:  # more info for stitching
    print(
        """Instructions for stitching:
        - Image stitching works by reading stage positions from the 'notes.txt' file generated during acquisition
        - Images MUST have:
            a. 'timepoint' substring in their names
            b. 'pos' or 'region' substring in their names
            c. channel substring(BF/GFP/RFP) in their names"""
    )

    user_check = input("Do you want to continue? (y/[n])") or "n"
    if user_check.casefold() == "n":
        print("Okay, bye!")
        exit()

top_dir = os.path.normpath(input("Enter the top directory with ALL acquisitions: "))

# %%
if action_flag != 3:
    bpf.oswalk_batchprocess_mip(main_dir=top_dir)

# # Stitching
if action_flag == 2:  # don't stitch and exit
    exit()

## Stitching
# all unique BF/GFP/RFP location gets added to the main_dir_list
main_dir_list = []
for root, subfolders, _ in os.walk(top_dir):
    if ("BF" in subfolders) or ("GFP" in subfolders) or ("RFP" in subfolders):
        main_dir_list.append(root)
main_dir_list = natsorted(main_dir_list)
print(f"Found these fish data:\n{main_dir_list}")

ch_names = ["BF", "GFP_mip", "RFP_mip"]

# main_dir = location of Directory containing ONE fish data
for main_dir in main_dir_list:
    print(f"Processing {main_dir}...")
    ch_2Dimg_flags, ch_2Dimg_paths, ch_2Dimg_lists = bpf.find_2D_images(main_dir)
    stage_coords = bpf.find_stage_coords_n_pixel_width_from_2D_images(ch_2Dimg_flags, ch_2Dimg_paths, ch_2Dimg_lists)
    global_coords_px = bpf.global_coordinate_changer(stage_coords)

    for ch_name, ch_2Dimg_flag, ch_2Dimg_path, ch_2Dimg_list in zip(
        ch_names, ch_2Dimg_flags, ch_2Dimg_paths, ch_2Dimg_lists
    ):
        if ch_2Dimg_flag:
            pos_max = bpf.pos_max
            print(f"Stitching {ch_name} images...")
            save_path_stitched_img = os.path.join(ch_2Dimg_path, f"{ch_name.casefold()}_stitched")
            save_path_stitched_edited_img = os.path.join(ch_2Dimg_path, f"{ch_name.casefold()}_stitched_bgsub_rescaled")
            bpf.check_create_save_path(save_path_stitched_img)
            bpf.check_create_save_path(save_path_stitched_edited_img)

            for i in tqdm(range(len(ch_2Dimg_list) // pos_max)):  # run once per timepoint
                # print(f"tp: {i+1}")
                img_list_per_tp = [0] * pos_max
                for j in range(0, pos_max):
                    loc = i * pos_max + j
                    # print(loc)
                    # save all pos images in 3D array
                    img = tiff.imread(os.path.join(ch_2Dimg_path, ch_2Dimg_list[loc]))
                    if len(img.shape) != 2:
                        print(f"{ch_2Dimg_list[loc]}: Image shape is not 2D... something is wrong. exiting...")
                        exit()
                    else:
                        img_list_per_tp[j] = img

                stitched_img, stitched_img_bgsub = bpf.img_stitcher_2D(global_coords_px, img_list_per_tp)
                # By default, the min/max intensities of the input image are stretched to the limits allowed by the image’s dtype, since in_range defaults to ‘image’ and out_range defaults to ‘dtype’:
                # stitched_img_bgsub_rescaled = skimage.exposure.rescale_intensity(stitched_img_bgsub) #produces images with pulsing mean intensity

                og_datatype = stitched_img_bgsub.dtype
                # use histogram matching using the first image
                if i == 0:  # set first stitched image as reference
                    ref_img_histogram = stitched_img_bgsub
                    stitched_img_bgsub_rescaled = stitched_img_bgsub
                else:  # match remaining images histogram to the first image
                    stitched_img_bgsub_rescaled = (
                        skimage.exposure.match_histograms(image=stitched_img_bgsub, reference=ref_img_histogram)
                    ).astype(og_datatype)

                skimage.io.imsave(
                    os.path.join(save_path_stitched_img, f"Timepoint{i + 1}_{ch_name}_stitched.png"),
                    stitched_img,
                    check_contrast=False,
                )  # save the stitched image
                skimage.io.imsave(
                    os.path.join(save_path_stitched_edited_img, f"Timepoint{i + 1}_{ch_name}_stitched.png"),
                    stitched_img_bgsub_rescaled,
                    check_contrast=False,
                )  # save the bg subtracted stitched image
print("Done! Processed images are in '<channelname>_mip' and '<channelname>_mip_bgsub_rescaled' folders")
# wait for user to close the window
input("Press Enter to close the program...")
exit()
