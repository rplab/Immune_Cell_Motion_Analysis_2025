# batchprocessing helper functions
# Author: Piyush Amitabh
# Created: Jan 15, 2024
# License: GNU GPL v3.0


import configparser
import os
import re
import shutil

import numpy as np
import skimage
import tifffile as tiff
from natsort import natsorted

## CONSTANT
# The pixel spacing in our LSM image is 1µm in the z axis, and  0.1625µm in the x and y axes.
ZD, XD, YD = 1, 0.1625, 0.1625

## Get global variables
pos_max = 0
img_h, img_w = 0, 0
find_scope_flag = 0
new_spacing = [0, 0, 0]


## Small Helping Functions
def check_create_save_path(save_path):
    """checks if the folder at 'save_path' alreay exists if not then it creates that folder"""
    save_path = os.path.normpath(save_path)
    if not os.path.exists(save_path):  # check if the dest exists
        print("Save path doesn't exist.")
        os.makedirs(save_path)
        print(f"Directory '{os.path.basename(save_path)}' created")
    else:
        print("Save path exists")


def remove_non_image_files(big_list, root_path):
    """removes dir and non-image(tiff) files from a list"""
    small_list = []
    for val in big_list:
        if os.path.isfile(os.path.join(root_path, val)):  # file check
            filename_list = val.split(".")
            # og_name = filename_list[0]
            ext = filename_list[-1]
            if ext == "tif" or ext == "tiff":  # img check
                small_list.append(val)
    return small_list


def remove_non_csv_files(big_list, root_path):
    """removes dir and non-csv files from files in the big_list at the root_path"""
    small_list = []
    for val in big_list:
        if os.path.isfile(os.path.join(root_path, val)):  # file check
            filename_list = val.split(".")
            # og_name = filename_list[0]
            ext = filename_list[-1]
            if ext == "csv":  # csv check
                small_list.append(val)
    return small_list


def reorder_files_by_pos_tp(file_list):
    """reoders the file_list by pos and tp"""
    file_list_arr = np.array(file_list)
    global pos_max
    raw_pos_arr = np.zeros_like(file_list_arr, dtype=int)
    raw_tp_arr = np.zeros_like(file_list_arr, dtype=int)
    for i, file_name in enumerate(file_list_arr):
        raw_pos_arr[i], raw_tp_arr[i] = find_pos_tp_in_filename(file_name)
    pos_max = np.max(raw_pos_arr)  # get pos_max
    ind = np.lexsort((raw_pos_arr, raw_tp_arr))  # Sort by tp, then by pos
    return file_list_arr[ind]


def find_pos_tp_in_filename(file_name):
    """finds the 'timepoint' and 'pos' in the filename (must be separated by '_')
    Returns int: file_name_pos, file_name_tp"""
    file_name_wo_ext = file_name.split(".")[0]
    file_name_split_list = file_name_wo_ext.split("_")
    file_name_pos, file_name_tp = -1, -1
    for substr in file_name_split_list:
        substr = substr.casefold()
        if "pos" in substr:
            file_name_pos = int(substr.removeprefix("pos"))
        if "timepoint" in substr:
            file_name_tp = int(substr.removeprefix("timepoint"))
    if (file_name_pos == -1) or (file_name_tp == -1):
        print(f"ParsingError: Couldn't find tp and pos from: {file_name}")
        print("Exiting...")
        exit()
    else:
        return (file_name_pos, file_name_tp)


def find_nearest_target_file(start_path, target):
    """finds the target file in the directory tree starting from start_path, returns the path of the file if found, else None"""
    found_file_path = None
    while True:
        if os.path.isfile(os.path.join(start_path, target)):
            # found target
            print(f"found {target} at:" + start_path)
            found_file_path = os.path.join(start_path, target)
            break
        elif os.path.dirname(start_path) == start_path:  # reached root
            # not found
            print(f"Warning: Couldn't find {target} file in the directory structure of {start_path}")
            break
        start_path = os.path.dirname(start_path)

    return found_file_path


def find_multi_subdir(subdir_path, subdir_name):
    """finds if there are multiple subdirectories at 'subdir_path' with the name 'subdir_name'"""
    sub_dir = os.listdir(subdir_path)
    multi_subdir_flag = False
    subdir_count = 0
    for sub in sub_dir:
        if subdir_name in sub.casefold():
            subdir_count += 1
            if subdir_count > 1:
                multi_subdir_flag = True
                break
    return multi_subdir_flag


def check_n_rename_old_imgname(save_name, path_name):
    """checks if the save_name is the old name and changes it to the new name which includes the subdir names"""
    old_name_flag = True
    channel_list = ["BF", "GFP", "RFP"]
    keywords = ["pos"] + channel_list

    for keyword in keywords:
        if keyword.casefold() in save_name.casefold():
            old_name_flag = False
            break

    if old_name_flag:
        save_name_list = path_name.split(os.sep)[-5:]
        for i in range(len(save_name_list)):
            # lowercase everthing in the list except the channel names
            if save_name_list[i] not in channel_list:
                save_name_list[i] = save_name_list[i].lower()
        # concatenate everything with a "_" in between
        save_name = "_".join(save_name_list)
    return save_name


# Important functions


def read_n_downscale_image(read_path, n):
    """Reads the image from read_path and downscales it by a factor of n in x and y dimensions."""
    print(f"Reading: {read_path}")
    img = tiff.imread(read_path)
    print(f"Shape of read image {img.shape}")
    if len(img.shape) == 2:  # 2 dimensional image, e.g. BF image
        # use a kernel of nxn, ds by a factor of n in x & y
        img_downscaled = skimage.transform.downscale_local_mean(img, (n, n))
    elif len(img.shape) == 3:  # image zstack
        # use a kernel of 1xnxn, no ds in z
        img_downscaled = skimage.transform.downscale_local_mean(img, (1, n, n))
    else:
        print("Can't process images with >3dimensions")
        return None
    # the downsampling algorithm produces float values,
    # so recast the ds image to the original datatype
    return np.round(img_downscaled).astype(img.dtype)


def single_acquisition_downsample(acq_path, new_trg_path, n):
    """downsamples the images in the Acquisition folder at the acq_path and saves them in the new_trg_path"""
    # Assuming the acq_path has the acquisition dir:
    # acq_path = Acquisition dir -> {fish1 dir, fish2 dir, etc.} + notes.txt
    files = os.listdir(acq_path)

    for filename in files:
        filename_list = filename.split(".")
        og_name = filename_list[0]  # first of list=name
        ext = filename_list[-1]  # last of list=extension
        if ext == "txt":  # copy text files
            shutil.copy(os.path.join(acq_path, filename), new_trg_path)
            print(f"copied text file: {filename}")

    # find if multiple `fish` folders are present at `acq_path`
    multi_fish_flag = find_multi_subdir(subdir_path=acq_path, subdir_name="fish")

    for root, subfolders, filenames in os.walk(acq_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            filename_list = filename.split(".")
            og_name = filename_list[0]  # first of list=name
            ext = filename_list[-1]  # last of list=extension

            if (ext == "tif" or ext == "tiff") and (
                not check_overflowed_stack(og_name)
            ):  # all tiff files, except overflowed stacks
                if multi_fish_flag:  # save the ds images in fish folder
                    fish_num_str = filepath[filepath.casefold().rfind("fish") + len("fish")]
                    try:
                        fish_num = int(fish_num_str)
                    except ValueError:
                        print(f"Error: Couldn't find fish number in {filepath}")
                        exit()
                    save_path = os.path.join(new_trg_path, "fish" + str(fish_num))
                    check_create_save_path(save_path)  # make fish num folders
                else:
                    save_path = new_trg_path

                if n == 1:  # no downscaling needed
                    img = tiff.imread(filepath)
                    save_name = f"{og_name.replace('_MMStack', '')}"
                    save_name = check_n_rename_old_imgname(save_name, root)
                    save_name = f"{save_name}.{ext}"
                    # shutil.copy(src=filepath, dst=os.path.join(save_path, save_name))
                    tiff.imwrite(os.path.join(save_path, save_name), data=img, compression="Deflate")
                    print(f"compressed image: {os.path.join(save_path, save_name)}")

                else:  # downscale by n
                    save_name = f"{og_name.replace('_MMStack', '')}"
                    save_name = check_n_rename_old_imgname(save_name, root)
                    save_name = f"{save_name}_ds.{ext}"

                    tiff.imwrite(
                        os.path.join(save_path, save_name),
                        read_n_downscale_image(read_path=filepath, n=n),
                    )


def find_2D_images(main_dir):
    """makes a list of all 2D img files by channel order 'BF, GFP, RFP' in the main_dir"""
    bf_flag, gfp_flag, rfp_flag = False, False, False  # 0 means not found, 1 mean found
    bf_path, gfp_mip_path, rfp_mip_path = "", "", ""
    bf_img_list, gfp_img_list, rfp_img_list = [], [], []

    for root, subfolders, filenames in os.walk(main_dir):
        for filename in filenames:
            # filepath = os.path.join(root, filename)
            # print(f'Reading: {filepath}')
            filename_list = filename.split(".")
            og_name = filename_list[0]  # first of list=name
            ext = filename_list[-1]  # last of list=extension

            if ext == "tif" or ext == "tiff":
                if (not bf_flag) and ("bf" in og_name.casefold()):  # find BF
                    print("BF images found at:" + root)
                    bf_path = root
                    bf_img_list = reorder_files_by_pos_tp(remove_non_image_files(natsorted(os.listdir(root)), root))
                    bf_flag = True
                elif "mip" in og_name.casefold():
                    if (not gfp_flag) and ("gfp" in og_name.casefold()):
                        print("GFP MIP images found at:" + root)
                        gfp_mip_path = root
                        gfp_img_list = reorder_files_by_pos_tp(
                            remove_non_image_files(natsorted(os.listdir(root)), root)
                        )
                        gfp_flag = True
                    elif (not rfp_flag) and ("rfp" in og_name.casefold()):
                        print("RFP MIP images found at:" + root)
                        rfp_mip_path = root
                        rfp_img_list = reorder_files_by_pos_tp(
                            remove_non_image_files(natsorted(os.listdir(root)), root)
                        )
                        rfp_flag = True
    if not bf_flag:
        print(f"No BF images found in {main_dir}")
    if not gfp_flag:
        print(f"No GFP MIP images found in {main_dir}")
    if not rfp_flag:
        print(f"No RFP MIP images found in {main_dir}")

    if not (bf_flag or gfp_flag or rfp_flag):
        print(f"No images found in {main_dir}. Exiting...")
        exit()
    return (
        [bf_flag, gfp_flag, rfp_flag],
        [bf_path, gfp_mip_path, rfp_mip_path],
        [bf_img_list, gfp_img_list, rfp_img_list],
    )


def find_3D_images(main_dir):
    """makes a list of all 3D img files by channel order 'GFP, RFP' in the main_dir"""
    # very general and robust code which looks for the files with those names instead of depending upon just the folder names
    # 3D images
    gfp_flag, rfp_flag = False, False  # 0 means not found, 1 mean found
    gfp_path, rfp_path = "", ""
    gfp_img_list, rfp_img_list = [], []
    for root, subfolders, filenames in os.walk(main_dir):
        for filename in filenames:
            # print(f'Reading: {filename}')
            # filepath = os.path.join(root, filename)
            # print(f'Reading: {filepath}')
            filename_list = filename.split(".")
            og_name = (filename_list[0]).casefold()  # first of list=name
            ext = (filename_list[-1]).casefold()  # last of list=extension
            if ext == "tif" or ext == "tiff":
                if ("bf" not in og_name) and ("mip" not in og_name):  # ignore BF and MIP
                    if ("gfp" in og_name) and (not gfp_flag):  # find GFP
                        print("GFP images found at:" + root)
                        gfp_path = root
                        gfp_img_list = reorder_files_by_pos_tp(
                            remove_non_image_files(natsorted(os.listdir(root)), root)
                        )
                        gfp_flag = True
                    elif ("rfp" in og_name) and (not rfp_flag):  # find RFP
                        print("RFP images found at:" + root)
                        rfp_path = root
                        rfp_img_list = reorder_files_by_pos_tp(
                            remove_non_image_files(natsorted(os.listdir(root)), root)
                        )
                        rfp_flag = True

    if not gfp_flag:
        print(f"*No* GFP images found in {main_dir}")
    if not rfp_flag:
        print(f"*No* RFP images found in {main_dir}")
    if not (rfp_flag or gfp_flag):
        print("Need at least one channel 3D images to work.. None found.. Exiting..")
        exit()
    return (
        [gfp_flag, rfp_flag],
        [gfp_path, rfp_path],
        [gfp_img_list, rfp_img_list],
    )


def find_lsm_scope(img_h, img_w):
    """Finds LSM Scope and downscaling factor automatically using image height and width.
    Returns:
    ds_factor_h = downscaling factor in height,
    ds_factor_w = downscaling factor in width"""

    global findscope_flag  # refer to the global var
    ds_factor_w, ds_factor_h = 1, 1

    if img_w == img_h:  # probably KLA LSM
        findscope_flag = 1
        ds_factor_h = 2048 // img_h
        ds_factor_w = 2048 // img_w
        r = 2048 % img_h
        if r > 0:  # implying downscaling factor is in fraction
            findscope_flag = 0
            print("Downscaling factor in fraction. Can't process automatically.")

    elif img_w > img_h:  # probably WIL LSM
        findscope_flag = 2
        ds_factor_h = 2160 // img_h
        ds_factor_w = 2560 // img_w
        if ds_factor_h != ds_factor_w:
            findscope_flag = 0
        r_h = 2160 % img_h
        r_w = 2560 % img_w
        if r_h > 0 or r_w > 0:  # implying downscaling factor is in fraction
            findscope_flag = 0
            print("Downscaling factor in fraction. Can't process automatically.")

    if findscope_flag == 1:
        print("LSM Scope used: KLA")
        print(f"Downscaling factor = {ds_factor_w}")
    elif findscope_flag == 2:
        print("LSM Scope used: WIL")
        print(f"Downscaling factor = {ds_factor_w}")

    if findscope_flag == 0:  # couldn't find scope, enter manually
        print("ERROR: Failed to determine LSM scope automatically.\nEnter manually")
        findscope_flag = int(input("Enter the scope used:\n1 - KLA LSM Scope\n2 - WIL LSM Scope\nInput (1/2): "))
        if findscope_flag == 1 or findscope_flag == 2:
            ds_factor_h = int(input("Enter the downscaling factor in height: "))
            ds_factor_w = int(input("Enter the downscaling factor in width: "))
        else:
            print("Fatal Error: Exiting")
            exit()
    return (ds_factor_h, ds_factor_w)


def find_stage_coords_n_pixel_width_from_2D_images(ch_flags, ch_paths, ch_img_lists):
    """Send channel flags and paths in the order [bf, gfp, rfp]"""
    # change global image_height and image_width
    global img_h, img_w, new_spacing
    # unpack variables
    bf_flag, gfp_flag, rfp_flag = ch_flags
    bf_path, gfp_mip_path, rfp_mip_path = ch_paths
    bf_img_list, gfp_img_list, rfp_img_list = ch_img_lists
    # find the nearest notes.txt
    config = configparser.ConfigParser()
    start_path = ""
    img_path = ""  # dummy
    # get start_path for search
    # get sample image to find scope and downscaling factor
    if bf_flag:
        start_path = bf_path
        img_path = os.path.join(bf_path, bf_img_list[0])
    elif gfp_flag:
        start_path = gfp_mip_path
        img_path = os.path.join(gfp_mip_path, gfp_img_list[0])
    elif rfp_flag:
        start_path = rfp_mip_path
        img_path = os.path.join(rfp_mip_path, rfp_img_list[0])

    # get sample image dimensions
    img = tiff.imread(img_path)
    img_h, img_w = img.shape[0], img.shape[1]
    (ds_h, ds_w) = find_lsm_scope(img_h, img_w)
    new_spacing = np.array([ZD, YD * ds_h, XD * ds_w])  # downscale x&y by n, skimage coords = z, y, x plane, row, col
    print(f"Pixel width (plane, row, col): {new_spacing}\n")

    # find the fish number from the image path
    fish_num = int(
        img_path[img_path.casefold().rfind("fish") + len("fish")]
    )  # find fish number starting from the img_name
    print(f"found fish_num = {fish_num}")

    target1 = "notes.txt"
    target2 = "Notes.txt"
    notes_path = find_nearest_target_file(start_path, target1) or find_nearest_target_file(start_path, target2)
    if notes_path is None:
        print("Error: Can't find notes.txt, Enter manually")
        notes_path = input("Enter complete path (should end with .txt): ")
    config.read(notes_path)

    # print(config.sections())
    abbrev = config.getfloat(f"Fish {fish_num} Region 1", "x_pos", fallback=False)
    if abbrev:
        # config_prop_list = ["x_pos", "y_pos", "z_pos"]
        config_prop_list = [
            "x_pos",
            "y_pos",
            "z_stack_start_pos",
        ]  # wil and kla stores in this format
        print(f"abbreviated props... reading {config_prop_list}")
    else:
        # config_prop_list = ["x_position", "y_position", "z_position"]
        config_prop_list = ["x_position", "y_position", "z_start_position"]
        print(f"not abbreviated props... reading {config_prop_list}")
    stage_coords = np.zeros(shape=(pos_max, 3))
    for i in range(1, pos_max + 1):
        for j, val in enumerate(config_prop_list):  # x/y/z axes
            stage_coords[i - 1][j] = config.getfloat(f"Fish {fish_num} Region {i}", val)
    print(f"Found stage_coords: \n{stage_coords}")
    return stage_coords


def find_stage_coords_n_pixel_width_from_3D_images(ch_flags, ch_paths, ch_img_lists):
    """Send channel flags and paths in the order [gfp, rfp]"""
    # change global image_height and image_width
    global img_h, img_w, new_spacing
    # unpack variables
    gfp_flag, rfp_flag = ch_flags
    gfp_stack_path, rfp_stack_path = ch_paths
    gfp_img_list, rfp_img_list = ch_img_lists
    # find the nearest notes.txt
    config = configparser.ConfigParser()
    start_path = ""
    img_path = ""  # dummy
    # get start_path for search
    # get sample image to find scope and downscaling factor
    if gfp_flag:
        start_path = gfp_stack_path
        img_path = os.path.join(start_path, gfp_img_list[0])
    elif rfp_flag:
        start_path = rfp_stack_path
        img_path = os.path.join(start_path, rfp_img_list[0])
    print(start_path)
    print(img_path)
    # get sample image dimensions
    img = (tiff.imread(img_path))[0]  # type: ignore #get one z slice
    if img.ndim != 2:
        print(f"ERROR: Image dimension is {img.ndim + 1}, expected 3")
        exit()
    img_h, img_w = img.shape[0], img.shape[1]
    (ds_h, ds_w) = find_lsm_scope(img_h, img_w)
    new_spacing = np.array([ZD, YD * ds_h, XD * ds_w])  # downscale x&y by n, skimage coords = z, y, x plane, row, col
    print(f"Pixel width (plane, row, col): {new_spacing}\n")

    # find the fish number from the image path
    fish_num = int(
        img_path[img_path.casefold().rfind("fish") + len("fish")]
    )  # find fish number starting from the img_name
    print(f"found fish_num = {fish_num}")

    target1 = "notes.txt"
    target2 = "Notes.txt"
    notes_path = find_nearest_target_file(start_path, target1) or find_nearest_target_file(start_path, target2)
    if notes_path is None:
        print("Error: Can't find notes.txt, Enter manually")
        notes_path = input("Enter complete path (should end with .txt): ")
    config.read(notes_path)

    # print(config.sections())
    abbrev = config.getfloat(f"Fish {fish_num} Region 1", "x_pos", fallback=False)
    if abbrev:
        # config_prop_list = ["x_pos", "y_pos", "z_pos"]
        config_prop_list = [
            "x_pos",
            "y_pos",
            "z_stack_start_pos",
        ]  # wil and kla stores in this format
        print(f"abbreviated props... reading {config_prop_list}")
    else:
        # config_prop_list = ["x_position", "y_position", "z_position"]
        config_prop_list = ["x_position", "y_position", "z_start_position"]
        print(f"not abbreviated props... reading {config_prop_list}")
    stage_coords = np.zeros(shape=(pos_max, 3))
    for i in range(1, pos_max + 1):
        for j, val in enumerate(config_prop_list):  # x/y/z axes
            stage_coords[i - 1][j] = config.getfloat(f"Fish {fish_num} Region {i}", val)
    print(f"Found stage_coords: \n{stage_coords}")
    return stage_coords


def global_coordinate_changer(stage_coords):
    """Parameters: stage_coords: read from notes.txt to stitch images
                new_spacing: contains pixel width of the images
    Returns: 2D np.array same shape as stage_coords
    """
    # original camera pixel width is 1µm in the z (leading!) axis, and  0.1625µm in the x and y axes.
    # for downsampled by 4, this will be orig pixel_width*4
    # To correct for the offset and stitch different acquisition positions together, note the following:
    # - regionprops generates local coords of objects in the image as `axis-0,1 and 2`: this is our `Local coords system`
    # - stage positions is saved by micromanager and obtained from the metadata as `XPositionUm`, similarly for y and z. this is a `global coords system` in the sense that it is same for all the images for a given sample
    # - Finally we define our `Global coordinates` to the data frames as **"y-ap-um", "x-dv-um", "z-lr-um"** where ap is anterior to posterior axis, dv is dorsal to ventral axis, lr is left-right axis (*don't know the direction*, doesn't matter as long as the local and global are in the same direction)

    # The direction of Global coords system is different in Klamath LSM and Willamette LSM:

    # Klamath LSM:
    # - Stage +Z <-> Global +Z_lr <-> Local axis-0: (so same direction) needs to be ensured that it is same for each acquisition
    # - Stage +Y <-> Global +Y_ap <-> Local axis-1: (so same direction) same for all acquisition
    # - Stage -X <-> Global +X_dv <-> Local axis-2: (so inverse direction) same for all acquisition

    # To resolve the inverse directions between global x and local axis-2, I am flipping the global coordinate system's x axis.

    # Willamete LSM:
    # - Stage +Z <-> Global +Z_lr <-> Local axis-0: (so same direction) needs to be verified for each acquisition
    # - Stage +Y <-> Global +Y_ap <-> Local axis-1: (so same direction) same for all acquisition
    # - Stage +X <-> Global -X_dv <-> Local axis-2: (so inverse direction) same for all acquisition

    # To resolve the flipped axes between global x/y and stage x/y, I am exchanging the global coordinate system's axis.

    stage_origin = stage_coords[0].copy()
    global_coords_um = stage_coords - stage_origin  # set first stage position as zero
    global_coords_um[:, 0] = global_coords_um[:, 0] * -1  # flip x axis
    # um to pixels: 1px = (1/pixel width) um
    # as the stage position is in x (col:axis-2), y(row:axis-1), z(plane:axis-0) format
    global_coords_px = global_coords_um / [
        new_spacing[2],
        new_spacing[1],
        new_spacing[0],
    ]
    return global_coords_px


def median_bg_subtraction(img_wo_bg_sub):
    """subtracts the median pixel intensity of the image from the image"""
    img_bg_sub = img_wo_bg_sub - np.median(img_wo_bg_sub.flatten())  # subtract median
    img_bg_sub[img_bg_sub < 0] = 0  # make all negative values zero
    return img_bg_sub


def check_overflowed_stack(filename):
    """return True if the 'filename' is a overflowed_stack else False"""
    num = filename[filename.casefold().rfind("mmstack_") + len("mmstack_")]
    return re.match(r"\d", num)


def oswalk_batchprocess_mip(main_dir):
    """Uses oswalk to find all 3D images in main_dir and create MIPs for GFP and RFP channels."""
    print("Finding Max Intensity Projections...")
    print(
        "Warning: This code ONLY works with single channel z-stack tiff images. It will give unpredictable results with >3 dimensions"
    )
    channel_names = ["GFP", "RFP"]
    for root, subfolders, filenames in os.walk(main_dir):
        for filename in filenames:
            # print(f'Reading: {filename}')
            filepath = os.path.join(root, filename)
            # print(f'Reading: {filepath}')
            filename_list = filename.split(".")
            og_name = filename_list[0]  # first of list=name
            ext = filename_list[-1]  # last of list=extension

            if (ext == "tif" or ext == "tiff") and (
                not check_overflowed_stack(og_name)
            ):  # tiff files which are not spilled-over stacks
                read_image = tiff.imread(filepath)
                if len(read_image.shape) == 3:  # check if 3D images
                    print(f"Processing MIP for: {filepath}")
                    arr_mip = np.max(read_image, axis=0)  # create MIP
                    # arr_mip_wo_bg_sub = np.max(read_image, axis=0) #create MIP
                    # arr_mip = median_bg_subtraction(arr_mip_wo_bg_sub)
                    for ch_name in channel_names:  # save mip array in right directory with correct channel name
                        if ch_name.casefold() in og_name.casefold():
                            dest = os.path.join(root, ch_name.casefold() + "_mip")
                            if not os.path.exists(dest):  # check if the dest exists
                                print("Write path doesn't exist.")
                                os.makedirs(dest)
                                print(f"Directory '{ch_name.casefold()}_mip' created")
                            # the downsampling algorithm produces float values,
                            # so recast the ds image to the original datatype
                            img_mip = np.round(arr_mip).astype(read_image.dtype)

                            if og_name.endswith("_MMStack"):  # remove 'MMStack' in saved name
                                save_name = og_name[: -len("_MMStack")] + "_mip." + ext
                            else:
                                save_name = og_name + "_mip." + ext
                            tiff.imwrite(os.path.join(dest, save_name), img_mip)


def img_stitcher_2D(global_coords_px, img_list):
    """accept a list of 2D images in img_list and use stage_coords read from notes.txt to stitch images
    Returns: 2D np.array containing the stitched image
    """
    if findscope_flag == 0:
        print("ERROR: Couldn't find the LSM scope")
        exit()
    # poses = np.shape(img_list)[0]
    og_datatype = img_list[0].dtype
    img_height = np.shape(img_list)[1]
    img_width = np.shape(img_list)[2]

    # stitched image ax0 is going down, ax1 is to the right
    ax0_offset, ax1_offset = [], []
    if findscope_flag == 2:  # wil lsm, stitch horizontally
        ax0_offset = global_coords_px[:, 0] * -1  # ax0 = -Global X_DV
        ax1_offset = global_coords_px[:, 1]  # ax1 = Global Y_AP
    elif findscope_flag == 1:  # kla lsm, stitch vertically
        ax0_offset = global_coords_px[:, 1]  # ax0 = Global Y_AP
        ax1_offset = global_coords_px[:, 0]  # ax1 = Global X_DV

    # find offset from min
    ax0_offset = np.ceil(ax0_offset - np.min(ax0_offset)).astype(int)
    ax1_offset = np.ceil(ax1_offset - np.min(ax1_offset)).astype(int)

    # find max of each axis
    ax0_max = img_height + np.max(ax0_offset)
    ax1_max = img_width + np.max(ax1_offset)

    # create empty stitched image
    stitched_image = np.zeros([ax0_max, ax1_max], dtype=og_datatype)  # rows-height, cols-width
    stitched_image_bg_sub = np.zeros_like(stitched_image)

    # bg subtract all images to be stitched
    img_list_bg_sub = []
    for img in img_list:
        img_list_bg_sub.append(median_bg_subtraction(img))

    # stitch images
    for i, (h0, w0) in enumerate(zip(ax0_offset, ax1_offset)):
        stitched_image[h0 : h0 + img_height, w0 : w0 + img_width] = img_list[i]
        stitched_image_bg_sub[h0 : h0 + img_height, w0 : w0 + img_width] = img_list_bg_sub[i]
    # return (np.round(stitched_image).astype(og_datatype), np.round(stitched_image_bg_sub).astype(og_datatype))
    # additional check
    if stitched_image.dtype != og_datatype:
        raise TypeError("Datatype is not preserved.. Something wrong.. Check code")
    return (stitched_image, stitched_image_bg_sub)


def img_stitcher_3D(global_coords_px, img_path_list, bg_sub=True, save_path=None, compression_type=None):
    """Accept a list of 3D image paths in img_path_list and use global_coords_px to stitch images.
    Returns: 3D np.array containing the stitched image.
    """
    if findscope_flag == 0:
        print("ERROR: Couldn't find the LSM scope")
        exit()

    # Read the first image to get the shape and datatype
    first_img = tiff.imread(img_path_list[0])
    if len(first_img.shape) != 3:
        print(f"{img_path_list[0]}: Image shape is not 3D... something is wrong. exiting...")
        exit()

    og_datatype = first_img.dtype
    img_height = first_img.shape[1]
    img_width = first_img.shape[2]
    z_width = [tiff.imread(img_path).shape[0] for img_path in img_path_list]

    # Determine offsets based on the scope type
    if findscope_flag == 2:  # wil lsm, stitch horizontally
        ax0_offset = global_coords_px[:, 0] * -1  # ax0 = -Global X_DV
        ax1_offset = global_coords_px[:, 1]  # ax1 = Global Y_AP
    elif findscope_flag == 1:  # kla lsm, stitch vertically
        ax0_offset = global_coords_px[:, 1]  # ax0 = Global Y_AP
        ax1_offset = global_coords_px[:, 0]  # ax1 = Global X_DV
    z_offset = global_coords_px[:, 2]  # ax2 = Global Z_lr

    # Find offset from min
    ax0_offset = np.ceil(ax0_offset - np.min(ax0_offset)).astype(int)
    ax1_offset = np.ceil(ax1_offset - np.min(ax1_offset)).astype(int)
    z_offset = np.ceil(z_offset - np.min(z_offset)).astype(int)

    # Find max of each axis
    ax0_max = img_height + np.max(ax0_offset)
    ax1_max = img_width + np.max(ax1_offset)
    z_max = np.max(z_width) + np.max(z_offset)

    # Create empty stitched image
    stitched_image = np.zeros([z_max, ax0_max, ax1_max], dtype=og_datatype)

    # Process and stitch images one by one
    for i, img_path in enumerate(img_path_list):
        img = tiff.imread(img_path)
        if len(img.shape) != 3:
            print(f"{img_path}: Image shape is not 3D... something is wrong. exiting...")
            exit()

        if bg_sub:
            img = median_bg_subtraction(img)

        z0, h0, w0 = z_offset[i], ax0_offset[i], ax1_offset[i]
        stitched_image[z0 : z0 + z_width[i], h0 : h0 + img_height, w0 : w0 + img_width] = img

    # Additional check
    if stitched_image.dtype != og_datatype:
        raise TypeError("Datatype is not preserved.. Something wrong.. Check code")

    # Save the stitched image if save_path is provided
    if save_path:
        tiff.imwrite(save_path, data=stitched_image, compression=compression_type, bigtiff=True)
        return None
    else:
        return stitched_image
