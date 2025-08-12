# 


import configparser
import os

import numpy as np
import pandas as pd
import plotly.express as px
import tifffile as tiff
from natsort import natsorted
from scipy.spatial import distance

## CONSTANT
# The pixel spacing in our LSM image is 1µm in the z axis, and  0.1625µm in the x and y axes.
ZD, XD, YD = 1, 0.1625, 0.1625

## Get global variables
POS_MAX = 0
img_h, img_w = 0, 0
find_scope_flag = 0
new_spacing = [0, 0, 0]


## Small Helping Functions
def find_stats(arr):
    print(f"Mean    {np.mean(arr):.2f}")
    print(f"Median  {np.median(arr):.2f}")
    print(f"Std     {np.std(arr):.2f}")
    print(f"COV (std/mean):{(np.std(arr) / np.mean(arr)):0.2f}")
    print(f"Min     {np.min(arr):.2f}")
    print(f"Max     {np.max(arr):.2f}\n")
    return ()


# check and create path
def check_create_save_path(save_path):
    save_path = os.path.normpath(save_path)
    if not os.path.exists(save_path):  # check if the dest exists
        print("Save path doesn't exist.")
        os.makedirs(save_path)
        print(f"Directory '{os.path.basename(save_path)}' created")
    else:
        print("Save path exists")


# removes dir and non-image(tiff) files from a list
def remove_non_image_files(big_list, root_path):
    small_list = []
    for val in big_list:
        if os.path.isfile(os.path.join(root_path, val)):  # file check
            filename_list = val.split(".")
            # og_name = filename_list[0]
            ext = filename_list[-1]
            if ext == "tif" or ext == "tiff":  # img check
                small_list.append(val)
    return small_list


# removes dir and non-csv files from a list
def remove_non_csv_files(big_list, root_path):
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
    """Reorders the file list by position and timepoint
    Parameters:
    file_list: list of file names
    Returns:
    file_list_arr: numpy array of file names sorted by position and timepoint
    """

    file_list_arr = np.array(file_list)
    global POS_MAX
    raw_pos_arr = np.zeros_like(file_list_arr, dtype=int)
    raw_tp_arr = np.zeros_like(file_list_arr, dtype=int)
    for i, file_name in enumerate(file_list_arr):
        file_name_list = file_name.split("_")
        for substr in file_name_list:
            substr = substr.casefold()
            if "pos" in substr:
                raw_pos_arr[i] = int(substr.removeprefix("pos"))
            if "timepoint" in substr:
                raw_tp_arr[i] = int(substr.removeprefix("timepoint"))
    POS_MAX = np.max(raw_pos_arr)  # get POS_MAX
    ind = np.lexsort((raw_pos_arr, raw_tp_arr))  # Sort by tp, then by pos
    return file_list_arr[ind]


def find_3D_images(main_dir, ofc_flag=False):
    """Finds 3D images in the main_dir
    Parameters:
    main_dir: directory to search for files
    ofc_flag: boolean, if True, search for offset corrected regionprops tables
    Returns:
    [gfp_flag, rfp_flag]: list of booleans indicating if the channel has sample images
    [gfp_path, rfp_path]: list of paths to the channel sample images
    [gfp_img_list, rfp_img_list]: list of lists of sample image file names for each channel
    """

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
            if (ext == "tif" or ext == "tiff") and not (rfp_flag or gfp_flag):  # just need one sample image
                if ("bf" not in og_name) and ("mip" not in og_name):  # ignore BF and MIP
                    if "gfp" in og_name:  # find GFP
                        print("GFP images found at:" + root)
                        gfp_path = root
                        gfp_img_list = reorder_files_by_pos_tp(
                            remove_non_image_files(natsorted(os.listdir(root)), root)
                        )
                        gfp_flag = True
                    elif "rfp" in og_name:  # find RFP
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


def find_2D_images_n_regionprop_csv(main_dir, ofc_flag):
    """Finds 2D images and regionprop csv files in the main_dir
    Parameters:
    main_dir: directory to search for files
    ofc_flag: boolean, if True, search for offset corrected regionprops tables
    Returns:
    [bf_flag, gfp_mip_flag, rfp_mip_flag]: list of booleans indicating if the channel has sample images
    [bf_path, gfp_mip_path, rfp_mip_path]: list of paths to the channel sample images
    [bf_img_list, gfp_img_list, rfp_img_list]: list of lists of sample image file names for each channel
    [gfp_csv_flag, rfp_csv_flag]: list of booleans indicating if the channel has regionprop csv files
    [gfp_csv_path, rfp_csv_path]: list of paths to the channel regionprop csv files
    [gfp_csv_list, rfp_csv_list]: list of lists of regionprop csv file names for each channel
    """
    # very general and robust code which looks for the files with those names instead of depending upon just the folder names
    # csv files
    gfp_csv_flag, rfp_csv_flag = False, False  # 0 means not found, 1 mean found
    gfp_csv_path, rfp_csv_path = "", ""
    gfp_csv_list, rfp_csv_list = [], []
    # make a list of all 2D img files by channel to get a sample image to find scope and downscaling factor
    bf_flag, gfp_mip_flag, rfp_mip_flag = False, False, False  # 0 means not found, 1 mean found
    bf_path, gfp_mip_path, rfp_mip_path = "", "", ""
    bf_img_list, gfp_img_list, rfp_img_list = [], [], []

    for root, subfolders, filenames in os.walk(main_dir):
        for filename in filenames:
            # print(f'Reading: {filename}')
            # filepath = os.path.join(root, filename)
            # print(f'Reading: {filepath}')
            filename_list = filename.split(".")
            og_name = filename_list[0]  # first of list=name
            ext = filename_list[-1]  # last of list=extension

            if ofc_flag:  # if true, read ofc tables
                ofc_condition = "_ofc" in og_name.casefold()
            else:  # if false, read non-ofc tables
                ofc_condition = "_ofc" not in og_name.casefold()

            # find csv with 'info' - regionprop tables and 'ofc' - offset corrected or not depending on above condition
            if (ext == "csv") and ("_info" in og_name.casefold()) and ofc_condition:
                if (not gfp_csv_flag) and ("gfp" in og_name.casefold()):
                    print(f"GFP regionprop with offset_corrected-{ofc_flag} csv found at: {root}")
                    gfp_csv_path = root
                    gfp_csv_list = reorder_files_by_pos_tp(remove_non_csv_files(natsorted(os.listdir(root)), root))
                    gfp_csv_flag = True
                elif (not rfp_csv_flag) and ("rfp" in og_name.casefold()):
                    print(f"RFP regionprop with offset_corrected-{ofc_flag} csv found at: {root}")
                    rfp_csv_path = root
                    rfp_csv_list = reorder_files_by_pos_tp(remove_non_csv_files(natsorted(os.listdir(root)), root))
                    rfp_csv_flag = True
            elif (ext == "tif" or ext == "tiff") and not (
                bf_flag or rfp_mip_flag or gfp_mip_flag
            ):  # just need one sample image
                if "bf" in og_name.casefold():  # find BF
                    print("BF images found at:" + root)
                    bf_path = root
                    bf_img_list = reorder_files_by_pos_tp(remove_non_image_files(natsorted(os.listdir(root)), root))
                    bf_flag = True
                elif "mip" in og_name.casefold():
                    if "gfp" in og_name.casefold():
                        print("GFP MIP images found at:" + root)
                        gfp_mip_path = root
                        gfp_img_list = reorder_files_by_pos_tp(
                            remove_non_image_files(natsorted(os.listdir(root)), root)
                        )
                        gfp_mip_flag = True
                    elif "rfp" in og_name.casefold():
                        print("RFP MIP images found at:" + root)
                        rfp_mip_path = root
                        rfp_img_list = reorder_files_by_pos_tp(
                            remove_non_image_files(natsorted(os.listdir(root)), root)
                        )
                        rfp_mip_flag = True
    if not (bf_flag or rfp_mip_flag or gfp_mip_flag):
        print(
            f"""*NO* sample image (BF/GFP mip/RFP mip) found in {main_dir}
        Need a sample image to find scope parameters. Exiting.."""
        )
        exit()
    if not gfp_csv_flag:
        print(f"*No* GFP regionprop offset_corrected-{ofc_flag} csv found in {main_dir}")
    if not rfp_csv_flag:
        print(f"*No* RFP regionprop offset_corrected-{ofc_flag} csv found in {main_dir}")
    if not (gfp_csv_flag or rfp_csv_flag):
        print("Need at least one channel csv tables to work.. None found.. Exiting..")

    return (
        [bf_flag, gfp_mip_flag, rfp_mip_flag],
        [bf_path, gfp_mip_path, rfp_mip_path],
        [bf_img_list, gfp_img_list, rfp_img_list],
        [gfp_csv_flag, rfp_csv_flag],
        [gfp_csv_path, rfp_csv_path],
        [gfp_csv_list, rfp_csv_list],
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


def read_all_regionprop_tables_save_in_dict(ch_names, ch_csv_flags, ch_csv_paths, ch_csv_lists):
    """Reads all regionprop csv files and saves them in a dictionary
    Parameters:
    ch_names: list of channel names in order [bf, gfp, rfp]
    ch_csv_flags: list of booleans indicating if the channel has csv files
    ch_csv_paths: list of paths to the channel csv files
    ch_csv_lists: list of lists of csv file names for each channel
    Returns:
    df_dict_gfp: dictionary with key=timepoint, val=regionprops table for GFP channel
    df_dict_rfp: dictionary with key=timepoint, val=regionprops table for RFP channel
    """
    # read all region_prop csv and save in dict
    df_dict_gfp, df_dict_rfp = {}, {}

    for ch_name, ch_csv_flag, ch_csv_path, ch_csv_list in zip(ch_names, ch_csv_flags, ch_csv_paths, ch_csv_lists):
        if ch_csv_flag:
            for i in range(len(ch_csv_list) // POS_MAX):  # run once per timepoint
                df_dict_per_tp = {}  # stores df for one tp
                for j in range(0, POS_MAX):
                    pos = j + 1
                    loc = i * POS_MAX + j
                    csv_name = ch_csv_list[loc]
                    print(f"Reading: {csv_name}")
                    df = pd.read_csv(os.path.join(ch_csv_path, csv_name))
                    df["pos"] = pos  # add a new column for storing positions
                    df_dict_per_tp[pos] = df

                if "gfp" == ch_name.casefold():  # concat all csv per timepoint and save in dict
                    df_dict_gfp[i] = pd.concat(df_dict_per_tp, ignore_index=True)
                    print("Save GFP timepoint now!")
                elif "rfp" == ch_name.casefold():  # concat all csv per timepoint and save in dict
                    df_dict_rfp[i] = pd.concat(df_dict_per_tp, ignore_index=True)
                    print("Save RFP timepoint now!")
                else:
                    print("Error: no GFP or RFP in channel name")
    return (df_dict_gfp, df_dict_rfp)


def find_stage_coords_n_pixel_width_from_3D_images(ch_flags, ch_paths, ch_img_lists):
    """Send channel flags and paths in the order [bf, gfp, rfp]"""
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
    while True:
        if os.path.isfile(os.path.join(start_path, target1)):
            # found
            print(f"found {target1} at:" + start_path)
            config.read(os.path.join(start_path, target1))
            break
        elif os.path.isfile(os.path.join(start_path, target2)):
            # found
            print(f"found {target2} at:" + start_path)
            config.read(os.path.join(start_path, target2))
            break

        if os.path.dirname(start_path) == start_path:  # reached root
            # not found
            print("Error: Can't find notes.txt, Enter manually")
            notes_path = input("Enter complete path (should end with .txt): ")
            config.read(notes_path)
            break
        start_path = os.path.dirname(start_path)
    # print(config.sections())
    abbrev = config.getfloat(f"Fish {fish_num} Region 1", "x_pos", fallback=False)
    if abbrev:
        # config_prop_list = ["x_pos", "y_pos", "z_pos"]
        config_prop_list = ["x_pos", "y_pos", "z_stack_start_pos"]  # wil and kla stores in this format
        print(f"abbreviated props... reading {config_prop_list}")
    else:
        # config_prop_list = ["x_position", "y_position", "z_position"]
        config_prop_list = ["x_position", "y_position", "z_start_position"]
        print(f"not abbreviated props... reading {config_prop_list}")
    stage_coords = np.zeros(shape=(POS_MAX, 3))
    for i in range(1, POS_MAX + 1):
        for j, val in enumerate(config_prop_list):  # x/y/z axes
            stage_coords[i - 1][j] = config.getfloat(f"Fish {fish_num} Region {i}", val)
    print(f"Found stage_coords: \n{stage_coords}")
    return stage_coords


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
    found_loc = img_path.casefold().rfind("fish")
    if found_loc == -1:
        print("ERROR: Couldn't find fish number in the image path")
        exit()
    fish_num = int(img_path[found_loc + len("fish")])  # find fish number starting from the img_name
    print(f"found fish_num = {fish_num}")
    target1 = "notes.txt"
    target2 = "Notes.txt"
    while True:
        if os.path.isfile(os.path.join(start_path, target1)):
            # found
            print(f"found {target1} at:" + start_path)
            config.read(os.path.join(start_path, target1))
            break
        elif os.path.isfile(os.path.join(start_path, target2)):
            # found
            print(f"found {target2} at:" + start_path)
            config.read(os.path.join(start_path, target2))
            break

        if os.path.dirname(start_path) == start_path:  # reached root
            # not found
            print("Error: Can't find notes.txt, Enter manually")
            notes_path = input("Enter complete path (should end with .txt): ")
            config.read(notes_path)
            break
        start_path = os.path.dirname(start_path)
    # print(config.sections())
    abbrev = config.getfloat(f"Fish {fish_num} Region 1", "x_pos", fallback=False)
    if abbrev:
        # config_prop_list = ["x_pos", "y_pos", "z_pos"]
        config_prop_list = ["x_pos", "y_pos", "z_stack_start_pos"]  # wil and kla stores in this format
        print(f"abbreviated props... reading {config_prop_list}")
    else:
        # config_prop_list = ["x_position", "y_position", "z_position"]
        config_prop_list = ["x_position", "y_position", "z_start_position"]
        print(f"not abbreviated props... reading {config_prop_list}")
    stage_coords = np.zeros(shape=(POS_MAX, 3))
    for i in range(1, POS_MAX + 1):
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
    global_coords_px = global_coords_um / [new_spacing[2], new_spacing[1], new_spacing[0]]
    return global_coords_px


# shift centroid
def shift_centroid(df, global_coords_px, ofc_flag):
    """Parameters: df: regionprops table with columns 'centroid_weighted-0,1,2' for each pos
    global_coords_px: 2D np.array with shape (POS_MAX, 3) containing global coordinates in pixels
    ofc_flag: boolean, if True, use offset corrected regionprops table
    Returns: None, modifies df in place by adding new columns for shifted centroids and global coordinates in pixels and micrometers
    """
    if findscope_flag == 0:
        print("ERROR: Couldn't find the LSM scope")
        exit()

    # Determine offsets based on the scope type
    elif findscope_flag == 2:  # wil lsm, stitch horizontally
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

    for i in range(POS_MAX):  # POS_MAX = 4, i=0, 1, 2, 3
        z0, h0, w0 = z_offset[i], ax0_offset[i], ax1_offset[i]
        df.loc[df.pos == i + 1, "shifted-centroid_weighted-0"] = df["centroid_weighted-0"] + z0
        df.loc[df.pos == i + 1, "shifted-centroid_weighted-1"] = df["centroid_weighted-1"] + h0
        df.loc[df.pos == i + 1, "shifted-centroid_weighted-2"] = df["centroid_weighted-2"] + w0
        if ofc_flag:
            df.loc[df.pos == i + 1, "shifted-centroid_weighted-0_ofc"] = df["centroid_weighted-0"] + z0
            df.loc[df.pos == i + 1, "shifted-centroid_weighted-1_ofc"] = df["centroid_weighted-1_ofc"] + h0
            df.loc[df.pos == i + 1, "shifted-centroid_weighted-2_ofc"] = df["centroid_weighted-2_ofc"] + w0

    if findscope_flag == 1:  # KLA LSM
        # add global coords in pixels: x-dv-px, y-ap-px, z-lr-px
        if ofc_flag:  # offset corrected regionprops used
            df["x-dv-px"] = df["shifted-centroid_weighted-2_ofc"]
            df["y-ap-px"] = df["shifted-centroid_weighted-1"]  # removed ofc
            df["z-lr-px"] = df["shifted-centroid_weighted-0_ofc"]
        else:
            df["x-dv-px"] = df["shifted-centroid_weighted-2"]
            df["y-ap-px"] = df["shifted-centroid_weighted-1"]
            df["z-lr-px"] = df["shifted-centroid_weighted-0"]

    elif findscope_flag == 2:  # WIL LSM
        # add global coords in pixels: x-dv-px, y-ap-px, z-lr-px
        if ofc_flag:
            df["x-dv-px"] = df["shifted-centroid_weighted-1_ofc"]
            df["y-ap-px"] = df["shifted-centroid_weighted-2"]  # removed ofc
            df["z-lr-px"] = df["shifted-centroid_weighted-0_ofc"]
        else:
            df["x-dv-px"] = df["shifted-centroid_weighted-1"]
            df["y-ap-px"] = df["shifted-centroid_weighted-2"]
            df["z-lr-px"] = df["shifted-centroid_weighted-0"]

    # make all the pixel values integers by dropping the decimal part
    df["x-dv-px"] = df["x-dv-px"].astype(int)
    df["y-ap-px"] = df["y-ap-px"].astype(int)
    df["z-lr-px"] = df["z-lr-px"].astype(int)

    # DONE: remove overlapped points
    for i in range(POS_MAX - 1):
        # drop points overlapped in the stitching
        # run only till the second last pos
        if findscope_flag == 1:  # kla lsm
            condition = (df["pos"] == (i + 1)) & (
                df["y-ap-px"] >= ax0_offset[i + 1]
            )  # pos is 1 indexed, but use hte next point of ax offset
            df.drop(df[condition].index, inplace=True)  # drop the rows
            df.reset_index(drop=True, inplace=True)  # reset the index

        elif findscope_flag == 2:  # wil lsm
            condition = (df["pos"] == (i + 1)) & (df["y-ap-px"] >= ax1_offset[i + 1])
            df.drop(df[condition].index, inplace=True)  # drop the rows
            df.reset_index(drop=True, inplace=True)  # reset the index

    # add global coords value in um
    df["x-dv-um"] = df["x-dv-px"] * new_spacing[2]
    df["y-ap-um"] = df["y-ap-px"] * new_spacing[1]
    df["z-lr-um"] = df["z-lr-px"] * new_spacing[0]
    return None


def find_nn_bw_timepoints(tp1, tp2):
    """
    finds nearest neighbor for all elements of tp1 in tp2

    Parameters:
    tpi (pd.DataFrame): dataframe containing the coords of the objects in the format 'x-dv-um, y-ap-um, z-lr-um'

    Returns:
    np.array: Array containing the nearest neighbors index in tp2 corresponding to each point in tp1
    """
    coords1 = [(tp1.loc[i, "x-dv-um"], tp1.loc[i, "y-ap-um"], tp1.loc[i, "z-lr-um"]) for i in tp1.index]
    coords2 = [(tp2.loc[i, "x-dv-um"], tp2.loc[i, "y-ap-um"], tp2.loc[i, "z-lr-um"]) for i in tp2.index]
    calc_dist = distance.cdist(
        coords1, coords2, metric="euclidean"
    )  # computes distance for all elements of coords1 to coords 2
    # by default argsort will sort along the last axis, here column so we get nn of tp1 in tp2
    nn_tp1_tp2 = np.argsort(calc_dist)[
        :, 0
    ]  # argsort sorts along last axis and [:,0] gets the first col corresponding to the nn
    return nn_tp1_tp2


# call the nn function to make a nn table for consecutive tp
def make_nn_list(df_dict):
    """Makes a list of nearest neighbors between consecutive timepoints in df_dict
    Parameters:
    df_dict: dictionary with key=timepoint, val=regionprops table for that timepoint
    Returns:
    nn_list_fwd: list of nearest neighbors going forward in time
    nn_list_bwd: list of nearest neighbors going backward in time
    """
    # df_dict should have keys as timepoints and values as regionprops tables for that time
    nn_list_fwd = []  # find nn going Forward in time
    nn_list_bwd = []  # find nn going Backward in time
    key = list(df_dict.keys())
    for i in range(len(key) - 1):
        nn_list_fwd.append(find_nn_bw_timepoints(tp1=df_dict[key[i]], tp2=df_dict[key[i + 1]]))
        nn_list_bwd.append(find_nn_bw_timepoints(tp1=df_dict[key[i + 1]], tp2=df_dict[key[i]]))
    return (nn_list_fwd, nn_list_bwd)


def search_track(val, track_dict):
    """Searches the entire track_dict for any occurence of val containing present object

    Parameters:
    val - tuple containing object information in (timepoint, S.no.);
    track_dict - dictionary containing all presently saved tracks

    Returns:
    int - track_id corresponding to successful search else 'None'
    """
    present_track_id = None
    for k in track_dict.keys():
        for i, v in enumerate(track_dict[k]):
            if val == v:
                present_track_id = k
    return present_track_id


# make a dictionary of keys=trackids, vals=lists of tuples (tp, obj_pos). Just save everything!


def make_track_dict(df_dict):
    """Makes a dictionary of tracks from df_dict containing regionprops tables for each timepoint
    Parameters:
    df_dict: dictionary with key=timepoint, val=regionprops table for that timepoint
    Returns:
    track_dict: dictionary with key=track_id, val=list of tuples (timepoint, object_position)
    """
    # initialize tracks
    track_id = 0
    track_dict = {}
    # make nearest neighbor list
    nn_list_fwd, nn_list_bwd = make_nn_list(df_dict)

    for fnum in range(len(nn_list_fwd)):
        for i in range(len(nn_list_fwd[fnum])):
            tp = list(df_dict.keys())[fnum]
            obj_pos = i
            present_obj = (tp, obj_pos)
            # check if present_obj already has a track
            p_track_id = search_track(present_obj, track_dict)

            if p_track_id is not None:  # if track already exists, implies present_obj has been already assigned
                if nn_list_bwd[fnum][nn_list_fwd[fnum][i]] == i:  # if good connection add next obj in track
                    next_obj = (list(df_dict.keys())[fnum + 1], nn_list_fwd[fnum][i])
                    track_dict[p_track_id].append(next_obj)

            else:  # if no present track start a new track
                track_id += 1
                track_dict[track_id] = []
                track_dict[track_id].append(present_obj)

                if nn_list_bwd[fnum][nn_list_fwd[fnum][i]] == i:  # if good connection add next obj in track
                    next_obj = (list(df_dict.keys())[fnum + 1], nn_list_fwd[fnum][i])
                    track_dict[track_id].append(next_obj)
    return track_dict


def get_track_info(track_dict, df_dict, prop_name):
    """uses track_dict and df_dict to get 'prop_name' for all tracks

    Parameters:
    track_dict: dictionary with key=track_id, val=(tp, obj_pos)
    df_dict: dictionary with key = tp, val= regionprop tables for that tp
    prop_name: string containing region_prop name

    Returns:
    pandas.DataFrame: Dataframe containing the 'prop_name' of all track points"""

    df_track_dict = {}  # dictionary containing dfs of track props
    for t_id, t_list in track_dict.items():
        nth_track_prop = []
        time_point = []
        for tp, obj_pos in t_list:
            nth_track_prop.append(df_dict[tp].loc[obj_pos, prop_name])  # row names is just integer so loc works
            time_point.append(tp)
        nth_track_df = pd.DataFrame({"track_id": t_id, prop_name: nth_track_prop, "time_point": time_point})
        df_track_dict[t_id] = nth_track_df
    return pd.concat(df_track_dict)  # ignore_index=True


# call this function to sever tracks by volume
def sever_tracks_by_vol(df):
    """Severs given df by volume variation in tracks, returning a df with new track_ids"""

    if "area" not in df:
        raise AttributeError("Passed DataFrame needs to have 3D-area property to sever by volume")

    copy_df = df.copy()
    new_tid = max(copy_df.track_id) + 1  # assign new track ids to severed shorter tracks
    orig_tid = list(set(copy_df.track_id))
    extended_tid = orig_tid

    for t_id in extended_tid:
        filt_df = copy_df[
            copy_df.track_id == t_id
        ]  # the number of elements in filt_df will be the same although some can get new t_id

        index_list = list(filt_df.index)

        for i in range(len(index_list) - 1):
            present_tid = filt_df.loc[index_list[i]].track_id
            next_tid = filt_df.loc[index_list[i + 1]].track_id

            if present_tid == next_tid:  # only makes sense to find the vol change within objects of same track
                present_vol = filt_df.loc[index_list[i]].area
                next_vol = filt_df.loc[index_list[i + 1]].area

                percent_increase = ((next_vol - present_vol) / present_vol) * 100
                percent_decrease = ((present_vol - next_vol) / next_vol) * 100

                if percent_increase > 150 or percent_decrease > 150:
                    print(f"Severed track: {t_id} after {index_list[i]} to {new_tid}")
                    for j in range((i + 1), len(index_list)):  # if true rename all following track ids
                        copy_df.loc[index_list[j], "track_id"] = new_tid  # make changes in orig df
                    extended_tid.append(new_tid)  # to recursively sever new tracks created
                    new_tid += 1
    return copy_df


# adds distance information to df and returns it
def find_dist_from_coords_df(df):
    copy_df = df.copy()
    track_id = list(set(copy_df.track_id))
    copy_df["distance_um"] = np.nan

    for t_id in track_id:
        filt_df = copy_df[
            copy_df.track_id == t_id
        ]  # the number of elements in filt_df will be the same although some can get new t_id
        index_list = list(filt_df.index)

        for i in range(len(index_list) - 1):
            present_coords = filt_df.loc[index_list[i], ["x-dv-um", "y-ap-um", "z-lr-um"]]
            next_coords = filt_df.loc[index_list[i + 1], ["x-dv-um", "y-ap-um", "z-lr-um"]]
            d_i_um = distance.euclidean(u=present_coords, v=next_coords)
            # copy_df.loc[index_list[i], 'distance_pixels'] = d_i_px
            copy_df.loc[index_list[i], "distance_um"] = d_i_um
    return copy_df


def make_df_super(track_dict, df_dict):
    """Makes a super dataframe containing all properties of all tracks in df_dict
    Parameters:
    track_dict: dictionary with key=track_id, val=list of tuples (timepoint, object_position)
    df_dict: dictionary with key=timepoint, val=regionprops table for that timepoint
    Returns:
    df_super: pandas.DataFrame containing all properties of all tracks
    df_super_sever: pandas.DataFrame containing all properties of all tracks severed by volume
    """

    # define super_prop_list as all columns except unnamed ones
    super_prop_list = [col for col in df_dict[0].columns if "unnamed" not in col.casefold()]
    # print(f"super_prop_list: {super_prop_list}")
    df_super = get_track_info(track_dict, df_dict, prop_name=super_prop_list[0])  # get first property df

    for i in range(1, len(super_prop_list)):  # get remaining properties and append them
        df1 = get_track_info(track_dict, df_dict, prop_name=super_prop_list[i])
        cols_to_use = df1.columns.difference(df_super.columns)
        df_super = pd.merge(df_super, df1[cols_to_use], left_index=True, right_index=True, how="outer")

    # sever super_df by volume before finding the distances
    df_super_sever = sever_tracks_by_vol(df_super)
    print("Track BEFORE sever:")
    find_stats(df_super.groupby("track_id").count().area)
    print("Track AFTER sever:")
    find_stats(df_super_sever.groupby("track_id").count().area)

    # adding distance information to the df_super
    df_super_w_dist = find_dist_from_coords_df(df_super)
    df_super_sever_w_dist = find_dist_from_coords_df(df_super_sever)
    return (df_super_w_dist, df_super_sever_w_dist)


def save_track_html(save_name, save_path, df_super, df_super_sever):
    track_html_filename = save_name + "_trajectory.html"
    track_html_filename_sever = save_name + "_trajectory_sever.html"

    fig1 = px.line_3d(df_super, z="x-dv-um", y="y-ap-um", x="z-lr-um", color="track_id", hover_data=["time_point"])
    fig1.update_layout(autosize=True, scene=dict(aspectmode="data"))
    fig1.write_html(os.path.join(save_path, track_html_filename))
    print(f"saved {track_html_filename}")

    fig2 = px.line_3d(
        df_super_sever, z="x-dv-um", y="y-ap-um", x="z-lr-um", color="track_id", hover_data=["time_point"]
    )
    fig2.update_layout(autosize=True, scene=dict(aspectmode="data"))
    fig2.write_html(os.path.join(save_path, track_html_filename_sever))
    print(f"saved  {track_html_filename_sever}")


def save_3dscatter_html(save_name, save_path, df_super):
    scatter_html_filename = save_name + "_3dscatter.html"
    if "rfp" in save_name.casefold():
        color = "lightcoral"
    elif "gfp" in save_name.casefold():
        color = "lightgreen"
    else:
        color = "cornflowerblue"
    fig1 = px.scatter_3d(df_super, z="x-dv-um", y="y-ap-um", x="z-lr-um")
    fig1.update_layout(autosize=True, scene=dict(aspectmode="data"))
    fig1.update_traces(marker=dict(size=6, color=color, line=dict(width=1, color="DarkSlateGrey")))
    fig1.write_html(os.path.join(save_path, scatter_html_filename))
    print(f"saved {scatter_html_filename}")
