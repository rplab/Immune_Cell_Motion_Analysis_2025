# %% [markdown]
# Author(s): Piyush Amitabh
#
# Details: this code downsamples, segments neutrophils and macrophages, saves regionprops to csv
#
# Created: March 31, 2022
#
# License: GNU GPL v3.0

# %% [markdown]
# ---
# %% [markdown]
# Created: Dec 06, 2022
# From: segmentation_and_regionprops_surface area_v2.ipynb
# Comments:
# - Batch processes downsampled images and saves their region properties
# - regionprops list updated for skimage-v0.19.2

# Updated: v9 on Feb 15, 2023
# Comments: made code os agnostic. As long as the original path is given correctly the code should work on windows or linux.
#
# date: mar 23, 23
# comment: extended for macrophages

# date: jan 7, 24
# comment: added a text metadata file which saves the script name used to generate the region props
# %% [markdown]
# ---

# date: feb 18, 24
# comment: change the segmentation of neuts to median + 7*std from abs value(1200),
# changed macs segmentation to use mean instead of median

# %% [markdown]
# ---
# list of regionprops updated for skimage-v0.19.2

# date: july 24, 24
# comment: added new surface area calculation which takes into account the anisotropic voxel size
# new segmentation method using morphological acwe
# changed name to show applicable for macs, neuts, tnfa



import os
from collections import defaultdict
from datetime import datetime

import custom_morphsnakes as cms
import numpy as np
import pandas as pd
import skimage as ski
import tifffile as tiff
from scipy import ndimage as ndi  # type: ignore

# %% [markdown]
# The pixel spacing in this dataset is 1µm in the z (leading!) axis, and  0.1625µm in the x and y axes.

n = 4  # downscaling factor in x and y
zd, xd, yd = 1, 0.1625, 0.1625  # zeroth dimension is z
ORIG_SPACING = np.array([zd, xd, yd])  # change to the actual pixel spacing from the microscope
NEW_SPACING = np.array([zd, xd * n, yd * n])  # downscale x&y by n

def active_threshold_volume_limit(img, max_volume=10**5):
    """
    Iteratively thresholds an image to ensure no objects exceed a specified volume.

    This function accepts a thresholded binary image (`binary_thresh`) and keeps iteratively
    thresholding it until there are no objects with a volume greater than the specified `max_volume`.

    Parameters:
    img (numpy.ndarray): The input image which needs to be thresholded.
    max_volume (int, optional): The maximum allowed volume for objects in the binary image. 
                                Objects with a volume greater than this value will be re-thresholded. 
                                Default is 10**5.

    Returns:
    numpy.ndarray: The thresholded binary image where no objects exceed the specified `max_volume`.
    """
    prefactor = 2
    img_median, img_std = np.median(img), np.std(img)
    init_binary_thresh = img > (img_median + prefactor*img_std)
    
    init_regionprop_df = pd.DataFrame(
        ski.measure.regionprops_table(ski.measure.label(init_binary_thresh), 
                                          properties=('label', 'slice', 'area'))
        )
    rethreshed_binary = init_binary_thresh.copy()
    df_bigobj = init_regionprop_df[init_regionprop_df.area > max_volume]
    i = 0
    while not df_bigobj.empty: #keep rethresholding until no big objects are found
        i += 1
        old_big_obj = len(df_bigobj)
        
        for _, row in df_bigobj.iterrows():
            obj_slice = row['slice']
            img_slice = img[obj_slice]               
            rethreshed_binary[obj_slice] = img_slice >  (np.median(img_slice) + prefactor*np.std(img_slice))
            
        rethreshed_labels = ski.measure.label(rethreshed_binary)
        regionprop_df = pd.DataFrame(
        ski.measure.regionprops_table(rethreshed_labels, 
                                          properties=('label', 'slice', 'area'))
        )
        df_bigobj = regionprop_df[regionprop_df.area > max_volume]
        new_big_obj = len(df_bigobj)
        
        if old_big_obj == new_big_obj:
            print(f'Iteration {i}')
            print(f'Old big obj = New big obj = {old_big_obj}')
            prefactor += 1
            print(f'No change in big objects, increasing prefactor to {prefactor}')
        elif i % 10 == 0: # if the loop has run for 10 iterations without any change in prefactor then increase by 1
            print(f'Loop has run for {i} iterations without any change in prefactor')
            print(f'This implies that number of big objects is oscillating between {old_big_obj} and {new_big_obj}')
            prefactor += 1
            print(f'Increasing prefactor to {prefactor}')
            
        if i > 100: #break the loop if it runs for too long
            print('**** WARNING: Breaking loop after 100 iterations ****')
            print(f'The number of big objects is oscillating between {old_big_obj} and {new_big_obj}')
            print(f'The number of big objects (vol>{max_volume}) in final result: {new_big_obj}')
            break
    
    return rethreshed_binary


def threshold_morphological_acwe(img, min_volume=10**3, max_volume=10**5, num_iter=10):
    """
    Applies a morphological Active Contour Without Edges segmentation to each image in a stack and saves the labels as tiff files.
    
    Parameters
    ----------
    img : numpy.ndarray
        The image stack to be segmented.
    min_volume : int, optional
        The minimum volume of objects to be retained. Default is 10**3.
    max_volume : int, optional
        The maximum volume of objects to be retained. Default is 10**5.
    num_iter : int, optional
        The number of iterations for the morphological ACWE algorithm. Default is 10.
        

    Returns
    -------
    numpy.ndarray: The segmented image stack.
    """
    
    rethreshed_binary = active_threshold_volume_limit(img, max_volume=max_volume)     #limit the max volume of objects
    binary_low_thresh_wo_small_objects = ski.morphology.remove_small_objects(rethreshed_binary, min_size=min_volume)
    labels_low_thresh = ski.measure.label(binary_low_thresh_wo_small_objects)
    regionprop_df = pd.DataFrame(
        ski.measure.regionprops_table(labels_low_thresh, properties=('label', 'slice'))
        )
    # add exception for empty regionprop_df
    if regionprop_df.empty:
        print("No objects found in the image")
        return np.zeros_like(img)
    
    # loop through all the labels and acwe on all the slices
    morphacwe_segmented = np.zeros_like(img)
    for _, row in regionprop_df.iterrows():        
        bbox_slice = row['slice']        
        lengths = [s.stop - s.start for s in bbox_slice]
        if any(length < 3 for length in lengths):
            continue  # Skip iteration if any dimension is less than 3
        
        img_slice = img[bbox_slice]  # slice the original img
        label_slice_img = labels_low_thresh[bbox_slice]  # type: ignore # slice the binary labels
        binary_select_label = (label_slice_img == row['label'])
        
        segmented = cms.morphological_chan_vese(img_slice, 
                                                num_iter=num_iter, 
                                                init_level_set=binary_select_label, 
                                                smoothing=1, lambda1=1, lambda2=4, 
                                                return_evolution=False, early_stop=True, early_stop_iter=3)
        morphacwe_segmented[bbox_slice] = segmented #assign the segmented slice to the full image
    
    #add a upper limit to the acwe
    binary_high_thresh = img > (np.median(img) + 7*np.std(img))
    segment_union = np.logical_or(morphacwe_segmented.astype(bool), binary_high_thresh)
    segment_union_wo_small_objects = ski.morphology.remove_small_objects(segment_union, min_size=min_volume)
    
    return segment_union_wo_small_objects


def acwe_segmentation_n_get_info_table(img_stack, **kwargs):
    """Segments and calculates properties of 3d image `img_stack` and returns regionprops table as pandas dataframe"""
    
    segmented_image = threshold_morphological_acwe(img_stack, **kwargs)
    
    # Define the column names
    prop_list = [
        "label",
        "slice",
        "bbox",
        "centroid",
        "centroid_weighted",
        "area",
        "equivalent_diameter_area",
        "intensity_mean",
    ]
    
    # Add exception for null segmented_image
    if np.sum(segmented_image) == 0:
        print("No objects found in the image")
        return pd.DataFrame()
    
    labels = ski.measure.label(segmented_image)
    info_table = pd.DataFrame(
        ski.measure.regionprops_table(
            labels,
            intensity_image=img_stack,
            properties=prop_list,
            cache=True
        )
    )#.set_index("label") keep label as column

    #add surface area, holes, sphericity
    find_surface_area(labels, info_table)
    return info_table

    
def find_surface_area(labels, info_table):
    """Uses marching cubes to find the surface area of the objects in info_table_filt
    Adds this user_surface_area and sphericity to the info_table_filt"""
    
    #define functions and variables needed for surface area calculation
    def binary_smooth_gaussian(binary_array: np.ndarray, sigma: float = 3) -> np.ndarray:
        vol = np.float64(binary_array) - 0.5
        return ndi.gaussian_filter(vol, sigma=sigma)
    
    def find_mesh_holes(faces):
        edge_count = defaultdict(int)
        # Step 1 & 2: Extract edges and count occurrences
        for face in faces:
            # Assuming triangular faces
            edges = [(face[i], face[(i+1) % 3]) for i in range(3)]
            for edge in edges:
                edge = tuple(sorted(edge))  # Sort the tuple to ensure uniqueness
                edge_count[edge] += 1
        # Step 3: Identify edges with single occurrence
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return len(boundary_edges)

    min_spacing = min(NEW_SPACING)
    scope_voxel_downscaling_factor = tuple(np.round(NEW_SPACING/min_spacing).astype(int))
    list_holes = np.zeros_like(info_table.index)
    list_surface_area = np.zeros_like(info_table.index)
    margin = 5

    for i, (selected_label, selected_slice) in enumerate(zip(info_table['label'], info_table['slice'])):
        bool_object = (labels == selected_label)
        voxel_slice = bool_object[selected_slice]
        voxel_slice_padded = np.pad(voxel_slice, margin, mode='constant') #very imp, else we get holes in the mesh
        volume_gaussian = binary_smooth_gaussian(voxel_slice_padded, sigma=scope_voxel_downscaling_factor) #almost same as sigma=1
        verts, faces, _normals, _values = ski.measure.marching_cubes(volume_gaussian, spacing=NEW_SPACING)
        list_holes[i] = find_mesh_holes(faces)
        surface_area_pixels = ski.measure.mesh_surface_area(verts, faces)
        list_surface_area[i] = surface_area_pixels

    info_table['holes'] = list_holes
    info_table['area_um'] = info_table['area']*np.prod(NEW_SPACING) #add area in um^3
    info_table['user_surface_area_um'] = list_surface_area
    info_table['sphericity'] = ((36*np.pi*(info_table['area_um'])**2)**(1/3))/info_table['user_surface_area_um']


def read_3d_img_save_props(img_path, save_path, csv_name):
    """
    Description:
    Reads 3D zstack images given by 'img_path', finds properties and saves them in 'csv_name' at 'save_path'
    """
    
    stack_full = tiff.imread(img_path)
    if len(stack_full.shape) != 3:  # return None if not a zstack
        return None

    print("Reading: " + img_path)
    info_table = acwe_segmentation_n_get_info_table(img_stack=stack_full, 
                                                    min_volume=10**3, max_volume=10**5, 
                                                    num_iter=10)

    # save info_table_filt
    if not os.path.exists(save_path):  # check if the save_path for region_props exist
        print("Save path doesn't exist")
        os.makedirs(save_path)
        print(f"{save_path} created..")
    else:
        print("save path exists..")
    info_table.to_csv(os.path.join(save_path, csv_name))
    print("Successfully saved: " + csv_name)


main_dir = input("Enter the Main directory containing ALL images to be segmented: ")

sub_dirs = ["RFP", "GFP"]  # as we can only segment in fluorescent channels
metadata_save_flag = False

print("Images need to be sorted in different directories by channel(BF/GFP/RFP).")
print("Run the sort_img_by_channels.py script before running this")

flag = input("Did you sort images by channel? (y/n)")
if flag.casefold().startswith("y"):
    print("Ok, starting segmentation")
else:
    print("Okay, bye!")
    exit()

# now do os walk then send all images to the segment function to starting segmentation
for root, subfolders, filenames in os.walk(main_dir):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        # print(f'Reading: {filepath}')
        filename_list = filename.split(".")
        og_name = filename_list[0]  # first of list=name
        ext = filename_list[-1]  # last of list=extension

        if ext == "tif" or ext == "tiff":  # only if tiff file
            # check image channel and create directory if it doesn't exist
            for sub in sub_dirs:
                if sub.casefold() in og_name.casefold():  # find the image channel
                    save_path = os.path.join(root, sub.casefold() + "_region_props") #"_region_props_acwe"
                    read_3d_img_save_props(img_path=filepath, save_path=save_path, csv_name=og_name + "_info.csv")
                    if metadata_save_flag is False:
                        with open(os.path.join(os.path.dirname(root), "region_props_metadata.txt"), "w") as file:
                            script_ver = f"code used to generate region prop tables: {os.path.basename(__file__)}"
                            time_stamp = f"time at which they were generated (first region prop generation time): {datetime.now()}"
                            file.write(f"{script_ver}\n{time_stamp}")
                        metadata_save_flag = True