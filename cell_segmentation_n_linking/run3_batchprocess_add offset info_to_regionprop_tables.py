# %% [markdown]
# Author(s): Piyush Amitabh
#
# Details: Adds pixel offset info and offset-corrected centroids to regionprop tables for segmented cells.
#
# Created: August 01, 2024
#
# License: GNU GPL v3.0

# %% [markdown]
# Reads **df_px_offset.csv** and **regionprop tables** of segmented objects to save:
# - offset information in the regionprop tables as additional columns `pixel_offset-1`, `pixel_offset-2`
# - the offset corrected centroids as `centroid_weighted-1_ofc`, `centroid_weighted-2_ofc`
#
# The new region prop tables are stored with '_ofc' appended to their filenames

# %%
import os

import batchprocessing_helper_functions as bpf
import pandas as pd
from natsort import natsorted
from tqdm.contrib import tzip

# %% [markdown]
# ## User input
# Prompt user for top-level directory
top_dir = input("Enter the path to the top-level directory: ")

# Display folder name and its contents
print(f"\nSelected folder: {top_dir}")
folder_contents = natsorted(os.listdir(top_dir))
print("Contents:")
for item in folder_contents:
    print(f" - {item}")

input("\nPress Enter to continue...")

# %%
# all unique BF location gets added to the main_dir_list
main_dir_list = []
for root, subfolders, _ in os.walk(top_dir):
    if "BF" in subfolders:
        main_dir_list.append(root)

print("\nImaging datasets found in main directory:")
for idx, dir_path in enumerate(main_dir_list, 1):
    print(f"{idx}. {dir_path}")

# %%
# Prompt user to enter number of positions per region, separated by commas
pos_max_input = input(
    f"Enter number of positions/regions of imaging per timepoint for each main directory (comma-separated, default={4}): "
)
if pos_max_input.strip() == "":
    pos_max_list = [4] * len(main_dir_list)
else:
    pos_max_list = [int(x.strip()) for x in pos_max_input.split(",")]
    # Ensure pos_max_list matches main_dir_list in length
    if len(pos_max_list) != len(main_dir_list):
        raise ValueError(
            f"Number of positions entered ({len(pos_max_list)}) does not match number of main directories found ({len(main_dir_list)}).\n"
            "Please enter the correct number of positions for each main directory, separated by commas."
        )

# %% [markdown]
# Prompt user for zstack channel names, default to ["GFP", "RFP"]
default_sub_names = ["GFP", "RFP"]
sub_names_input = input(f"Enter zstack channel names separated by commas (default={default_sub_names}): ")
if sub_names_input.strip() == "":
    sub_names = default_sub_names
else:
    sub_names = [x.strip() for x in sub_names_input.split(",")]

# %%
for main_dir, pos_max in tzip(
    main_dir_list, pos_max_list
):  # main_dir = location of Directory containing ONE fish data \n(this must contain BF/MIPs):
    # make a list of all csv files by channel for offset addition
    gfp_csv_flag, rfp_csv_flag = False, False  # 0 means not found, 1 mean found
    gfp_csv_path, rfp_csv_path = "", ""
    gfp_csv_list, rfp_csv_list = [], []
    for root, subfolders, filenames in os.walk(main_dir):
        for filename in filenames:
            # print(f'Reading: {filename}')
            filepath = os.path.join(root, filename)
            # print(f'Reading: {filepath}')
            filename_list = filename.split(".")
            og_name = filename_list[0]  # first of list=name
            ext = filename_list[-1]  # last of list=extension

            # find csv with 'info' - regionprop tables but not already 'ofc' - offset corrected
            if (
                (ext == "csv")
                and ("_info" in og_name.casefold())
                and ("_ofc" not in og_name.casefold())
                and ("old" not in root.casefold())
            ):
                if (not gfp_csv_flag) and ("gfp" in og_name.casefold()):
                    print("GFP regionprop csv found at:" + root)
                    gfp_csv_path = root
                    gfp_csv_list = bpf.reorder_files_by_pos_tp(
                        bpf.remove_non_csv_files(natsorted(os.listdir(root)), root)
                    )
                    gfp_csv_flag = True
                elif (not rfp_csv_flag) and ("rfp" in og_name.casefold()):
                    print("RFP regionprop csv found at:" + root)
                    rfp_csv_path = root
                    rfp_csv_list = bpf.reorder_files_by_pos_tp(
                        bpf.remove_non_csv_files(natsorted(os.listdir(root)), root)
                    )
                    rfp_csv_flag = True
    if not gfp_csv_flag:
        print(f"No GFP regionprop csv found in {main_dir}")
    if not rfp_csv_flag:
        print(f"No RFP regionprop csv found in {main_dir}")

    # read regionprop csv
    # get start_path for search
    if gfp_csv_flag:
        start_csv_path = gfp_csv_path
    elif rfp_csv_flag:
        start_csv_path = rfp_csv_path
    else:
        print("Error: Cannot find any regionprop csv")
        exit()
    df_px_offset_read = pd.read_csv(bpf.find_nearest_target_file(start_path=start_csv_path, target="df_px_offset.csv"))

    # add centroid ofc corrected
    ch_flag_list = [gfp_csv_flag, rfp_csv_flag]
    regionprop_path_list = [gfp_csv_path, rfp_csv_path]
    ch_csv_list = [gfp_csv_list, rfp_csv_list]

    for k, ch_flag in enumerate(ch_flag_list):
        all_csv_list = ch_csv_list[k]
        ch_name = sub_names[k]
        regionprop_path = regionprop_path_list[k]

        if ch_flag:
            save_path = os.path.join(os.path.dirname(regionprop_path), f"{ch_name}_region_props_ofc")
            if not os.path.exists(save_path):  # check if the dest exists
                print("Save path doesn't exist.")
                os.makedirs(save_path)
                print(f"Directory '{ch_name}_region_props_ofc' created")
            else:
                print("Save path exists")

            for i, csv_name in enumerate(all_csv_list):  # run once per csv
                # get offsets
                csv_pos, csv_tp = bpf.find_pos_tp_in_filename(csv_name)
                filt = (df_px_offset_read["timepoint"] == csv_tp) & (df_px_offset_read["pos"] == csv_pos)
                r_offset = df_px_offset_read["row_offset"][filt].values[0]
                c_offset = df_px_offset_read["col_offset"][filt].values[0]
                # read and edit regionprop csv
                df_regionprops = pd.read_csv(os.path.join(regionprop_path, csv_name))

                if df_regionprops.empty:  # if empty csv, just save #add offset and save
                    print(f"Empty csv: {csv_name}")
                    # add empty columns "pixel_offset-1" and "pixel_offset-2"
                    # df_regionprops["pixel_offset-1"] = pd.Series(dtype="float64")
                    # df_regionprops["pixel_offset-2"] = pd.Series(dtype="float64")
                    df_regionprops.to_csv(os.path.join(save_path, csv_name.split(".")[0] + "_ofc.csv"))
                    continue

                df_regionprops["pixel_offset-1"] = r_offset
                df_regionprops["pixel_offset-2"] = c_offset

                # simple addition works as the "offset" saved is the shift required to register offset_img with ref_img
                # r_offset/c_offset is shift needed along ax0/ax1 axis of 2D image eqv. to ax1/ax2 of 3d image
                df_regionprops["centroid_weighted-1_ofc"] = df_regionprops["centroid_weighted-1"] + r_offset
                df_regionprops["centroid_weighted-2_ofc"] = df_regionprops["centroid_weighted-2"] + c_offset
                df_regionprops.to_csv(os.path.join(save_path, csv_name.split(".")[0] + "_ofc.csv"))
print("All done!")
