# Author: Piyush Amitabh
# Created: September 01, 2023
# Last Modified: September 01, 2024
# Details: This code downsamples and sorts images by channel in the specified directory.
# License: GNU GPL v3.0

import os
import shutil
from pathlib import Path

import batchprocessing_helper_functions as bpf

# %%
action_flag = 0
while action_flag == 0:
    print("Downsample all tiff images inside acquisition folder by n, enter n=1 for simple copy")
    print("Sort images ONLY if the original filename has the channel name(BF/GFP/RFP)")
    action_flag = int(
        input(
            """Do you want to:
                        1. Downsample and sort images by channel (default)
                        2. Only Downsample images
                        3. Only Sort images by channel\n"""
        )
        or "1"
    )
    if action_flag == 1 or action_flag == 2 or action_flag == 3:
        break
    else:
        action_flag = 0
        print("Invalid value: Re-Enter")

# %%
# get user input for source and dest for action_flag 1 and 2
if action_flag == 1 or action_flag == 2:
    src, trg = "", ""
    while src == trg or src == "" or trg == "":
        src = os.path.normpath(
            input("Enter the Parent folder for original images (should contain 'Acquisition' folders): ")
        )
        trg = os.path.normpath(input("Enter the Destination folder: "))
        if src == trg:
            print("Parent and Destination folders cannot be empty or have the same location. Re-Enter..")
else:  # only for action_flag 3 (sort by channel)
    trg_path = os.path.normpath(input("Enter the Parent folder for images (image names should contain channel name): "))

# %%
if action_flag != 3:  # Downsample
    # n is the downscaling factor in x and y, change it accordingly.
    n = int(input("Enter downscaling factor for x and y dimensions (default=4):") or "4")
    if n < 1:
        print("User Error: downscaling factor MUST be a positive integer. Exiting")
        exit()
    elif n == 1:
        print("Downscaling factor is 1, no downscaling will be done but images will be compressed.")
        print(
            "Images will be saved in '.tif' format with Lossless 'Deflate' compression. This should reduce file size by ~ 50%."
        )
        user_confirm = input("Do you want to continue? ([y]/n):") or "y"
        if user_confirm.casefold() != "y":
            print("Exiting..")
            exit()
    # single_fish_flag is used to find if single acquisitions have single fish or not
    # single_fish_input = input("Is there ONLY 1 fish per Acquisition? ([y]/n):") or "y"
    # if single_fish_input.casefold() not in ("y", "n"):
    #     print("User Error: Need to enter 'y' or 'n'. Exiting")
    #     exit()
    # single_fish_flag = True if single_fish_input.casefold() == "y" else False

    new_folder_name = f"{os.path.split(src)[-1]}_downsampled_n{n}"
    trg_path = os.path.join(trg, new_folder_name)
    bpf.check_create_save_path(trg_path)
    print("Downsampling images..")
    # oswalk to find all acquisition folders
    for root, subfolders, filenames in os.walk(src):
        for sub in subfolders:  # separate by acquisitions
            if "acquisition" in sub.casefold():
                single_trg_path = root.replace(src, trg_path)

                multi_acq_folder_flag = bpf.find_multi_subdir(subdir_path=root, subdir_name="acquisition")
                if multi_acq_folder_flag:
                    # print("Multiple acquisition folders found at the same level...")
                    single_trg_path = os.path.join(single_trg_path, sub)

                single_acq_path = os.path.join(root, sub)
                path = Path(single_trg_path)  # copy entire folder structure
                path.mkdir(parents=True, exist_ok=True)

                bpf.single_acquisition_downsample(single_acq_path, single_trg_path, n)

if action_flag != 2:  # Sort by channel
    print("Sorting images by channel..")
    # sort images by channel
    sub_dirs = ["BF", "GFP", "RFP"]
    read_dir = trg_path

    for root, subfolders, filenames in os.walk(read_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            # print(f'Reading: {filepath}')
            filename_list = filename.split(".")
            og_name = filename_list[0]  # first of list=name
            ext = filename_list[-1]  # last of list=extension

            if ext == "tif" or ext == "tiff":  # only if tiff file
                # check image channel and create directory if it doesn't exist
                for sub in sub_dirs:
                    if sub.casefold() in og_name.casefold():
                        dest = os.path.join(root, sub)
                        if not os.path.exists(dest):  # check if the subdir exists
                            print("New path doesn't exist.")
                            os.makedirs(dest)
                            print(f"Directory '{sub}' created")
                        shutil.move(filepath, dest)  # move files
print("Successfully completed all tasks!")
# wait for user to close the window
input("Press Enter to close the program...")
exit()
