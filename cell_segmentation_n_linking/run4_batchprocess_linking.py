# %% [markdown]
# Author(s): Piyush Amitabh
#
# Details: this code reads the csv files created by "segmentation_and_regionprops" and performs linking on it
#
# Created: April 21, 2022
#
# License: GNU GPL v3.0

# %% [markdown]
# Modified: May 9, 2022
#
# Note: Added forward, backward check in time for NN and purged non-invertible NN functions

# %% [markdown]
# Modified: May 20, 2022
#
# Note: Added graphs for keeping track of region-props in a trajectory

# %% [markdown]
# Modified: June 22, 2022
#
# Note: cleaned code, added msd calculation for different time intervals

# %% [markdown]
# Modified: June 28, 2022
#
# Note: read all position tables, and perform 3d volumetric stiching
# also cleaned code

# %% [markdown]
# Modified: July 07, 2022
#
# Note: processes new timelapse data, using the absolute reference positions given by the stage

# %% [markdown]
# Modified: Dec 15, 2022
#
# Note: new axis system
# - Added the Global coordinates to the data frames as **"y-ap-um", "x-dv-um", "z-lr-um"** where ap is anterior to posterior axis, dv is dorsal to ventral axis, lr is left-right axis(*don't know the direction*, doesn't matter as long as the local and global are in the same direction)
#
# - Changed NN algorithm to find the nearest neighbor based on spatial distance rather than pixel distance

# %% [markdown]
# Modified: Feb 22, 2023
#
# Note: Removed all extra code. Purpose is to batchprocess and save df_super for each dataset

# %% [markdown]
# Modified: May 23, 2023
#
# Note: changed to read data from the offset corrected caudal fin cut images
# added the code for finding the right notes.txt file for finding the stage coords and other automations

# %% [markdown]
# ----

# %% [markdown]
# Many parts rewritten and automated
#
# Modified: v11 Jan11, 2024
# Comment: moved all the functions to a separate script, now the ofc/non-ofc can be read by a simple change of flag
import os

import analysis_n_linking_helper_functions as alf
from natsort import natsorted

# User input
print("\nThis code will link the objects whose data is stored in regionprops tables")
print("Generate regionprops tables before running this and set the right flag OFC/non-ofc to use")
print("Output will be saved in the 'linking_results' folder created inside the regionprops folder")
top_dir = input("\nEnter top directory containing ALL days of imaging data: ")
print("This directory contains\n", natsorted(os.listdir(top_dir)))

OFC_FLAG = (input("Are regionprops offset corrected? [y]/n: ") or "y").strip().lower() == 'y'
print(f"OFC_FLAG: {OFC_FLAG}")

#%%
gfp_cell_name = input("Enter GFP cell name [neutrophil]:") or "neutrophil"
rfp_cell_name = input("Enter RFP cell name [macrophage]:") or "macrophage"

# all unique GFP/RFP location gets added to the main_dir_list
main_dir_list = []
for root, subfolders, _ in os.walk(top_dir):
    if ("GFP" in subfolders) or ("RFP" in subfolders):
        main_dir_list.append(root)
print("Detected sub-folders with data:\n", main_dir_list)

# NOT AN INPUT: this is the channel order for all boolean flag lists
ch_names = ["GFP", "RFP"]
# ---
# start loop
for main_dir in main_dir_list:
    print(f"Reading: {main_dir}..")

    (
        ch_img_flags,
        ch_img_paths,
        ch_img_lists,
        ch_csv_flags,
        ch_csv_paths,
        ch_csv_lists,
    ) = alf.find_2D_images_n_regionprop_csv(main_dir, ofc_flag=OFC_FLAG)
    ## Read all tables
    df_dict_gfp, df_dict_rfp = alf.read_all_regionprop_tables_save_in_dict(
        ch_names, ch_csv_flags, ch_csv_paths, ch_csv_lists
    )
    # all the region-prop csv tables have been saved as dict values:
    # - all pos per tp concatenated and saved as single df
    # - key corresponding to timepoints (starting from 0)

    ## Shift to Global Coords
    stage_coords = alf.find_stage_coords_n_pixel_width_from_2D_images(ch_img_flags, ch_img_paths, ch_img_lists)

    global_coords_px = alf.global_coordinate_changer(stage_coords)

    # read all df
    for ch_name, ch_csv_flag in zip(ch_names, ch_csv_flags):
        if ch_csv_flag:
            if "gfp" == ch_name.casefold():
                for tp in df_dict_gfp.keys():
                    alf.shift_centroid(df_dict_gfp[tp], global_coords_px, OFC_FLAG)
            elif "rfp" == ch_name.casefold():
                for tp in df_dict_rfp.keys():
                    alf.shift_centroid(df_dict_rfp[tp], global_coords_px, OFC_FLAG)
    # Find NN
    # find nearest neighbors between consecutive timepoints
    for ch_name, ch_csv_flag in zip(ch_names, ch_csv_flags):
        if ch_csv_flag:
            if "gfp" == ch_name.casefold():
                track_dict_gfp = alf.make_track_dict(df_dict_gfp)
            elif "rfp" == ch_name.casefold():
                track_dict_rfp = alf.make_track_dict(df_dict_rfp)
    for ch_name, ch_csv_flag in zip(ch_names, ch_csv_flags):
        if ch_csv_flag:
            if "gfp" == ch_name.casefold():
                print(f"GFP: We have {len(track_dict_gfp)} tracks!")
                track_length = []
                for i in track_dict_gfp.keys():
                    track_length.append(len(track_dict_gfp[i]))
                print("GFP: track length characteristics")
                alf.find_stats(track_length)
            elif "rfp" == ch_name.casefold():
                print(f"RFP: We have {len(track_dict_rfp)} tracks!")
                track_length = []
                for i in track_dict_rfp.keys():
                    track_length.append(len(track_dict_rfp[i]))
                print("RFP: track length characteristics")
                alf.find_stats(track_length)

    ## df Super Sever
    # make df_super and save
    for ch_name, ch_csv_flag, ch_csv_path in zip(ch_names, ch_csv_flags, ch_csv_paths):
        if ch_csv_flag:
            if "gfp" == ch_name.casefold():
                df_super_gfp, df_super_sever_gfp = alf.make_df_super(
                    track_dict=track_dict_gfp, df_dict=df_dict_gfp
                )
                # save in "linking_results" in the same directory as the regionprops path
                gfp_results_save_path = os.path.join(ch_csv_path, "linking_results")
                alf.check_create_save_path(gfp_results_save_path)
                df_super_gfp.to_csv(os.path.join(gfp_results_save_path, "df_super_gfp.csv"))
                df_super_sever_gfp.to_csv(os.path.join(gfp_results_save_path, "df_super_sever_gfp.csv"))
            elif "rfp" == ch_name.casefold():
                df_super_rfp, df_super_sever_rfp = alf.make_df_super(
                    track_dict=track_dict_rfp, df_dict=df_dict_rfp
                )
                rfp_results_save_path = os.path.join(ch_csv_path, "linking_results")
                alf.check_create_save_path(rfp_results_save_path)
                df_super_rfp.to_csv(os.path.join(rfp_results_save_path, "df_super_rfp.csv"))
                df_super_sever_rfp.to_csv(os.path.join(rfp_results_save_path, "df_super_sever_rfp.csv"))
    # Now df_super has:
    # - all properties mentioned above
    # - severed tracks by volume variation
    # - distance information
    # and it is saved in the linking_results folder

    # Save trajectory HTML
    for ch_name, ch_csv_flag, ch_csv_path in zip(ch_names, ch_csv_flags, ch_csv_paths):
        if ch_csv_flag:
            if "gfp" == ch_name.casefold():
                alf.save_track_html(
                    save_name=gfp_cell_name,
                    save_path=gfp_results_save_path,
                    df_super=df_super_gfp,
                    df_super_sever=df_super_sever_gfp,
                )

            elif "rfp" == ch_name.casefold():
                alf.save_track_html(
                    save_name=rfp_cell_name,
                    save_path=rfp_results_save_path,
                    df_super=df_super_rfp,
                    df_super_sever=df_super_sever_rfp,
                )
