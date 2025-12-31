import numpy as np
from scipy.spatial import distance
from skimage.measure import label
from skimage.morphology import dilation, remove_small_holes
from tqdm.auto import tqdm


def get_single_frame_velocity_track(df_super_concat, delta_t_min):
    """finds the single frame velocity and speed in the passed df_dictionary containing coordinates"""
    coord_cols = ["x-dv-um", "y-ap-um", "z-lr-um"]
    velocity_cols = [
        "velocity_x_um-per-min",
        "velocity_y_um-per-min",
        "velocity_z_um-per-min",
    ]

    for fish_id in tqdm(
        df_super_concat.fish_id.unique(), desc="fish_id"
    ):  # filter by fish_id
        filt_one_fish = df_super_concat[df_super_concat.fish_id == fish_id]
        for t_id in filt_one_fish.track_id.unique():  # filter by track_id
            unique_fish_tid_filt = (df_super_concat.fish_id == fish_id) & (
                df_super_concat.track_id == t_id
            )
            all_coords_per_tid = df_super_concat[unique_fish_tid_filt][
                coord_cols
            ].values
            velocity_arr = np.vstack(
                (
                    np.diff(all_coords_per_tid, axis=0) / delta_t_min,
                    [np.nan, np.nan, np.nan],
                )
            )
            df_super_concat.loc[unique_fish_tid_filt, velocity_cols] = velocity_arr
    # add speed
    df_super_concat["speed_um-per-min"] = np.linalg.norm(
        df_super_concat[velocity_cols], axis=1
    )


def add_distance_um_px(df_super_concat, delta_t_min, pixel_spacing):
    """give pixel_spacing in um, in the order (plane, row, column)"""
    distance_cols = ["distance_um", "distance_px"]
    # Remove existing distance columns if they exist
    for col in distance_cols:
        if col in df_super_concat.columns:
            df_super_concat.drop(columns=[col], inplace=True)
    df_super_concat["distance_um"] = (
        (df_super_concat["velocity_x_um-per-min"] * delta_t_min) ** 2
        + (df_super_concat["velocity_y_um-per-min"] * delta_t_min) ** 2
        + (df_super_concat["velocity_z_um-per-min"] * delta_t_min) ** 2
    ) ** 0.5

    df_super_concat["distance_px"] = (
        (df_super_concat["velocity_x_um-per-min"] * delta_t_min / pixel_spacing[2]) ** 2
        + (df_super_concat["velocity_y_um-per-min"] * delta_t_min / pixel_spacing[1])
        ** 2
        + (df_super_concat["velocity_z_um-per-min"] * delta_t_min / pixel_spacing[0])
        ** 2
    ) ** 0.5


# Add Subtracks information
def add_subtrack_info(df_super_concat, d_thresh_col="distance_px>mean_radius/2"):
    """based on frame-to-frame displacement threshold using the `d_thresh_col`,
    breaks a track into multiple subtracks. adds subtrack_id info to the df_super_concat"""
    for fish_id in tqdm(
        df_super_concat.fish_id.unique(), position=0, desc="fish_id"
    ):  # filter by fish_id
        filt_one_fish = df_super_concat[df_super_concat.fish_id == fish_id]
        present_max_track_id = filt_one_fish.track_id.max()

        for t_id in filt_one_fish.track_id.unique():  # filter by track_id
            unique_fish_tid_filt = (df_super_concat.fish_id == fish_id) & (
                df_super_concat.track_id == t_id
            )
            df_filt = df_super_concat[unique_fish_tid_filt]

            # remove upto 3 tp holes
            filled_motion_trajectory = dilation(
                remove_small_holes(df_filt[d_thresh_col].values, area_threshold=3)
            )
            # get subtrack id numbers
            motile_subtrack_num, num_motile_segments = label(
                filled_motion_trajectory, background=False, return_num=True
            )
            non_motile_subtrack_num = label(filled_motion_trajectory, background=True)
            # renumber non_motile_subtrack_num to start from the max motile_subtrack_num
            non_motile_subtrack_num[non_motile_subtrack_num != 0] += num_motile_segments  # type: ignore
            # get final track id array
            sub_track_id_arr = motile_subtrack_num + non_motile_subtrack_num  # type: ignore

            # make unique
            sub_track_id_arr = sub_track_id_arr + present_max_track_id
            present_max_track_id = sub_track_id_arr.max()

            df_super_concat.loc[unique_fish_tid_filt, "subtrack_id"] = sub_track_id_arr


def get_single_frame_velocity_subtrack(df_super_concat, delta_t_min):
    """finds the single frame velocity and speed in the passed df_dictionary containing coordinates"""
    coord_cols = ["x-dv-um", "y-ap-um", "z-lr-um"]
    velocity_cols = [
        "velocity_x_um-per-min",
        "velocity_y_um-per-min",
        "velocity_z_um-per-min",
    ]
    # Remove existing velocity columns if they exist
    for col in velocity_cols:
        if col in df_super_concat.columns:
            df_super_concat.drop(columns=[col], inplace=True)

    for fish_id in tqdm(
        df_super_concat.fish_id.unique(), desc="fish_id"
    ):  # filter by fish_id
        filt_one_fish = df_super_concat[df_super_concat.fish_id == fish_id]
        for st_id in filt_one_fish.subtrack_id.unique():  # filter by subtrack_id
            unique_fish_stid_filt = (df_super_concat.fish_id == fish_id) & (
                df_super_concat.subtrack_id == st_id
            )
            all_coords_per_stid = df_super_concat[unique_fish_stid_filt][
                coord_cols
            ].values
            velocity_arr = np.vstack(
                (
                    np.diff(all_coords_per_stid, axis=0) / delta_t_min,
                    [np.nan, np.nan, np.nan],
                )
            )
            df_super_concat.loc[unique_fish_stid_filt, velocity_cols] = velocity_arr
    # add speed
    df_super_concat["speed_um-per-min"] = np.linalg.norm(
        df_super_concat[velocity_cols], axis=1
    )


def add_velocity_cosine_similarity_subtrack(df_super_concat):
    df_super_concat["velocity_cosine_similarity"] = np.nan  # initialize array
    velocity_cols = [
        "velocity_x_um-per-min",
        "velocity_y_um-per-min",
        "velocity_z_um-per-min",
    ]
    for fish_id in tqdm(
        df_super_concat.fish_id.unique(), position=0, desc="fish_id"
    ):  # filter by fish_id
        filt_one_fish = df_super_concat[df_super_concat.fish_id == fish_id]
        for st_id in filt_one_fish.subtrack_id.unique():  # filter by subtrack_id
            unique_fish_stid_filt = (df_super_concat.fish_id == fish_id) & (
                df_super_concat.subtrack_id == st_id
            )
            index_list = list(df_super_concat[unique_fish_stid_filt].index)
            # if index list has repeated elements raise value error
            if len(df_super_concat.index.to_list()) != len(
                set(df_super_concat.index.to_list())
            ):
                raise ValueError(
                    "Index list has repeated elements. check and reset index before continuing."
                )

            for i in range(len(index_list) - 1):
                present_v = (
                    df_super_concat.loc[index_list[i], velocity_cols]
                ).to_numpy(dtype=np.float32)
                next_v = (
                    df_super_concat.loc[index_list[i + 1], velocity_cols]
                ).to_numpy(dtype=np.float32)
                # length of the vectors must be one row and 3 columns, check this
                if (present_v.shape != (3,)) or (next_v.shape != (3,)):
                    print(f"present_v: {present_v}, next_v: {next_v}")
                    raise ValueError("velocity vector shape is not (3,)")
                present_v_norm = np.linalg.norm(present_v, axis=0)  # same as speed
                next_v_norm = np.linalg.norm(next_v, axis=0)
                # print(f"present_v: {present_v}, next_v: {next_v}")
                # print(f"present_v_norm: {present_v_norm}, next_v_norm: {next_v_norm}")

                norm_product = present_v_norm * next_v_norm

                if norm_product == 0:
                    cosine_similarity = np.nan
                    angle_rad = np.nan
                    angle_deg = np.nan
                    df_super_concat.loc[index_list[i], "velocity_cosine_similarity"] = (
                        np.nan
                    )
                else:
                    cosine_similarity = np.dot(present_v, next_v) / norm_product
                    if (cosine_similarity > 1.00001) or (
                        cosine_similarity < -1.00001
                    ):  # check if within floating point error of [-1, 1]
                        print(f"cosine_similarity is BAD {cosine_similarity}")
                        raise ValueError("cosine_similarity is not between -1 and 1")
                    else:
                        cosine_similarity = np.clip(cosine_similarity, -1, 1)
                    angle_rad = np.arccos(cosine_similarity)
                    angle_deg = np.degrees(angle_rad)

                df_super_concat.loc[index_list[i], "velocity_cosine_similarity"] = (
                    cosine_similarity
                )
                df_super_concat.loc[index_list[i], "velocity_angle_rad"] = angle_rad
                df_super_concat.loc[index_list[i], "velocity_angle_deg"] = angle_deg

            df_super_concat.loc[unique_fish_stid_filt, "persistence"] = df_super_concat[
                unique_fish_stid_filt
            ]["velocity_cosine_similarity"].mean()


def add_spatial_extent_subtrack(df_super_concat):
    """adds max pairwise distance or spatial extent of all the track ids in df_super"""
    coord_cols = ["x-dv-um", "y-ap-um", "z-lr-um"]
    for fish_id in tqdm(
        df_super_concat.fish_id.unique(), position=0, desc="fish_id"
    ):  # filter by fish_id
        filt_one_fish = df_super_concat[df_super_concat.fish_id == fish_id]
        for st_id in filt_one_fish.subtrack_id.unique():  # filter by subtrack_id
            unique_fish_stid_filt = (df_super_concat.fish_id == fish_id) & (
                df_super_concat.subtrack_id == st_id
            )
            all_coords_per_stid = df_super_concat[unique_fish_stid_filt][
                coord_cols
            ].values

            if all_coords_per_stid.shape[0] == 1:  # only one object in subtrack_id
                max_pdist = np.nan
            else:
                max_pdist = np.max(distance.pdist(all_coords_per_stid, "euclidean"))
            df_super_concat.loc[unique_fish_stid_filt, "spatial_extent_um"] = max_pdist


# add anisotropy_index
def add_anisotropy_index(df_super_concat):
    """adds anisotropy_index column where anisotropy = sqrt(Vap**2/(Vdv**2 + Vlr**2))"""
    df_super_concat["anisotropy_index"] = df_super_concat[
        "velocity_y_um-per-min"
    ].abs() / (
        (
            df_super_concat["velocity_x_um-per-min"] ** 2
            + df_super_concat["velocity_z_um-per-min"] ** 2
        )
        ** 0.5
    )
