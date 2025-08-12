# %% [markdown]
# Author(s): Piyush Amitabh
# 
# Details: batchprocess the pixel offsets and save the offsets in a csv file, adds offset info to region props tables and corrects the offsets in images before stitching them
# 
# Created: May 22, 2023
# 
# License: GNU GPL v3.0
# 
# More info:  
# (1) Most Code taken from:
# a. find_n_fix_px_offset.ipynb
# b. batchprocess_stitch_using_notes.py
# 
# (2) directory locations must be given in the input cells below. The code doesn't ask for any other user feedback. For interactive version of the code, check the above script/notebooks.
# 

# %% [markdown]
# Updated: May 08, 2024
# 
# Comment: refactored pixel offset code to use functions defined in batchprocessing_functions_v3

# %%
import os

import batchprocessing_helper_functions as bpf
import numpy as np
import pandas as pd
import skimage

# from PIL import Image, TiffTags
import tifffile as tiff
from natsort import natsorted
from tqdm import tqdm

# %% [markdown]
# # User input
# 

# %%
top_dir = input("Enter the Main directory containing ALL data that needs to be Offset corrected: ")

print(f'This contains following folders {natsorted(os.listdir(top_dir))}')

# %%
# all unique BF location gets added to the main_dir_list
main_dir_list = []
for root, subfolders, _ in os.walk(top_dir):
    if "BF" in subfolders:
        main_dir_list.append(root)
main_dir_list = natsorted(main_dir_list)
print(f"Found these fish data BF folders:\n{main_dir_list}")

# %% [markdown]
# ## Execute the following steps per fish data
# 1. Find and Fix Px Offset. Also saves the pixel offet in df_px_offet.csv
# 2.  Stitch images. this finds the nearest notes.txt per fish then stitches them

# %%
ch_names = ["BF", "GFP_mip", "RFP_mip"]

# %%
for main_dir in main_dir_list:  # main_dir = location of Directory containing ONE fish data
    print(f"Processing {main_dir}...")
    ch_2Dimg_flags, ch_2Dimg_paths, ch_2Dimg_lists = bpf.find_2D_images(main_dir)
    stage_coords = bpf.find_stage_coords_n_pixel_width_from_2D_images(ch_2Dimg_flags, ch_2Dimg_paths, ch_2Dimg_lists)
    global_coords_px = bpf.global_coordinate_changer(stage_coords)

    # 1. Find and Save Px Offset    
    ## Find Offset using masks
    # find offset and save in df_px_offset.csv
    # Only need BF images
    bf_flag = ch_2Dimg_flags[0]
    bf_path = ch_2Dimg_paths[0]
    bf_img_list = ch_2Dimg_lists[0]
    
    if not bf_flag:
        print("Error: Cannot find the pixel offset without BF images")
        exit()
    pos_max = bpf.pos_max
    print("---------------------------------------------------")
    print("** Finding offset in BF images... **")
    save_path = os.path.join(os.path.dirname(bf_path), "df_px_offset.csv")
    row_offset, col_offset = np.zeros_like(bf_img_list), np.zeros_like(bf_img_list)
    tp_list, pos_list = np.zeros_like(bf_img_list), np.zeros_like(bf_img_list)

    for i in range(pos_max):
        row_offset[i], col_offset[i] = 0, 0
        tp_list[i] = 1
        pos_list[i] = i + 1

    for i in tqdm(range(1, len(bf_img_list) // pos_max)):  # run once per timepoint
        for j in range(pos_max):
            loc = i * pos_max + j  # gives the location in the list
            prev_loc = (i - 1) * pos_max + j
            prev_row_offset, prev_col_offset = (
                row_offset[prev_loc],
                col_offset[prev_loc],
            )
            # ref_img = tiff.imread(os.path.join(bf_path, bf_img_list[prev_loc])) #using previous img as ref doesn't work
            prev_img = tiff.imread(os.path.join(bf_path, bf_img_list[prev_loc]))

            # use corrected version of previous image as the reference image
            tform = skimage.transform.SimilarityTransform(translation=(prev_col_offset, prev_row_offset))
            ref_img_uint = skimage.util.img_as_uint(
                skimage.exposure.rescale_intensity(
                    skimage.transform.warp(prev_img, tform.inverse)
                )
            )  # rescale float and change dtype to uint16#corrected previous img
            # generate mask correspongding to this image
            mask_i = np.ones_like(prev_img)  # initial mask same as ref img
            mask = np.bool_(skimage.transform.warp(mask_i, tform.inverse))

            tp_list[loc] = i + 1
            pos_list[loc] = j + 1
            bf_offset_image = tiff.imread(
                os.path.join(bf_path, bf_img_list[loc])
            )  # read present image
            # shift, error, diffphase = skimage.registration.phase_cross_correlation(reference_img[j], bf_offset_image) #without mask
            shift = skimage.registration.phase_cross_correlation(
                reference_image=ref_img_uint,
                moving_image=bf_offset_image, 
                reference_mask=mask)[0] #just get the first value
            # old skimage version<22
            # shift, error, diffphase = skimage.registration.phase_cross_correlation(ref_img_uint,
            #                 bf_offset_image, reference_mask=mask, return_error='always')
            # old skimage version<21
            # shift, error = skimage.registration.phase_cross_correlation(
            #     ref_img_uint,
            #     bf_offset_image,
            #     reference_mask=mask,
            #     return_error=True,
            # )
            (row_offset[loc], col_offset[loc]) = shift  #shift is in row, col <-> (y, x) of a std graph
    df_px_offset = pd.DataFrame(
        {
            "timepoint": tp_list,
            "pos": pos_list,
            "row_offset": row_offset,
            "col_offset": col_offset,
        }
    )
    df_px_offset.to_csv(save_path)

    #2. Fix Offset in all 2D images
    ## find the nearest df_px_offset.csv
    df = pd.read_csv(bpf.find_nearest_target_file(start_path=bf_path, target="df_px_offset.csv"))
    
    for ch_name, ch_2Dimg_flag, ch_2Dimg_path, ch_2Dimg_list in zip(
        ch_names, ch_2Dimg_flags, ch_2Dimg_paths, ch_2Dimg_lists
    ):
        if ch_2Dimg_flag:
            pos_max = bpf.pos_max
            print("---------------------------------------------------")
            print(f"** Fixing offset in {ch_name} images... **")
            save_path = os.path.join(
                ch_2Dimg_path, ch_name.casefold() + "_offset_corrected"
            )
            if not os.path.exists(save_path):  # check if the dest exists
                print("Save path doesn't exist.")
                os.makedirs(save_path)
                print(f"Directory '{ch_name.casefold()}_offset_corrected' created")
            else:
                print("Save path exists")

            for i in tqdm(range(len(ch_2Dimg_list)//pos_max)):  # run once per timepoint
                tp = i + 1
                img_list_per_tp = [0] * pos_max
                for j in range(0, pos_max):
                    loc = i * pos_max + j
                    p = j + 1
                    single_2d_img = tiff.imread(os.path.join(ch_2Dimg_path, ch_2Dimg_list[loc]))  # save a single 2D image
                    # get its value from the df_px_offset
                    filt = (df["pos"] == p) & (df["timepoint"] == tp)
                    r_offset = df["row_offset"][filt].values[0]
                    c_offset = df["col_offset"][filt].values[0]
                    
                    #linear translate image in the graph axis form (x,y) <-> (ax0, ax1)
                    findscope_flag = bpf.findscope_flag
                    if findscope_flag==1: #kla
                        tform = skimage.transform.SimilarityTransform(translation=(c_offset, 0))#only column _offset
                    elif findscope_flag==2: #wil
                        tform = skimage.transform.SimilarityTransform(translation=(0, r_offset))#only row_offset

                    # transform while preserving range and change dtype back to uint16
                    warped_img_uint = skimage.transform.warp(single_2d_img, tform.inverse, 
                                                             preserve_range=True).astype(single_2d_img.dtype) 
                    tiff.imwrite(
                        os.path.join(save_path, ch_2Dimg_list[loc]), warped_img_uint
                    )  # save the image

    # 3. Stitch offset corrected images

    for ch_name, ch_2Dimg_flag, ch_2Dimg_path, ch_2Dimg_list in zip(
        ch_names, ch_2Dimg_flags, ch_2Dimg_paths, ch_2Dimg_lists
    ):
        ch_2Dimg_path_ofc = os.path.join(ch_2Dimg_path, f"{ch_name.casefold()}_offset_corrected") #use this path for stitching
        if ch_2Dimg_flag:
            pos_max = bpf.pos_max
            print("---------------------------------------------------")
            print(f"** Stitching {ch_name} images... **")
            save_path_stitched_img = os.path.join(ch_2Dimg_path_ofc, ch_name.casefold() + "_ofc_stitched")
            save_path_stitched_edited_img = os.path.join(ch_2Dimg_path_ofc, f"{ch_name.casefold()}_ofc_stitched_bgsub_rescaled")
            bpf.check_create_save_path(save_path_stitched_img)
            bpf.check_create_save_path(save_path_stitched_edited_img)

            for i in tqdm(range(len(ch_2Dimg_list) // pos_max)):  # run once per timepoint
                # print(f"tp: {i+1}")
                img_list_per_tp = [0] * pos_max
                for j in range(0, pos_max):
                    loc = i * pos_max + j
                    # print(loc)
                    # save all pos images in 3D array
                    img = tiff.imread(os.path.join(ch_2Dimg_path_ofc, ch_2Dimg_list[loc]))
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
                # or use this for conversion to uint16
                # skimage.util.img_as_uint(skimage.exposure.rescale_intensity(stitched_img_bgsub_rescaled, in_range=(0, 65535), out_range='dtype'))
                

                skimage.io.imsave(
                    os.path.join(save_path_stitched_img, f"Timepoint{i+1}_{ch_name}_stitched.png"),
                    stitched_img,
                    check_contrast=False,
                )  # save the stitched image
                skimage.io.imsave(
                    os.path.join(save_path_stitched_edited_img, f"Timepoint{i+1}_{ch_name}_stitched.png"),
                    stitched_img_bgsub_rescaled,
                    check_contrast=False,
                )  # save the bg subtracted stitched image

