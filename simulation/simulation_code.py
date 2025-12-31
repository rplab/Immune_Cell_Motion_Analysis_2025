import analysis_functions_simplified as af
import numpy as np
import pandas as pd

# Constants
# ORIG_PIXEL_WIDTH_UM = (1, 0.1625, 0.1625)  # (z, y, x in μm per pixel)
NEW_PIXEL_WIDTH_UM = (1.0, 0.65, 0.65)  # (z, y, x in μm per pixel)


# Functions
def random_unit_vectors(n):
    """Generate n uniformly distributed unit vectors on sphere."""
    phi = np.random.uniform(0, 2 * np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return np.column_stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
    )


def apply_detector_limits(sim_df, res_xy=0.5, res_z=1.0, add_noise=True, seed=None):
    """
    Apply detector resolution: quantization + localization noise.
    Parameters:
        sim_df (pd.DataFrame): simulated dataframe with 'x-dv-um', 'y-ap-um', 'z-lr-um' columns
        res_xy (float): resolution in xy (um)
        res_z (float): resolution in z (um)
        add_noise (bool): whether to add Gaussian noise before quantization
        seed (int|None): random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    out = sim_df.copy()

    if add_noise:
        sigma_xy = res_xy / 2.35
        sigma_z = res_z / 2.35
        n = len(out)
        out["x-dv-um"] += np.random.normal(0, sigma_xy, n)
        out["y-ap-um"] += np.random.normal(0, sigma_xy, n)
        out["z-lr-um"] += np.random.normal(0, sigma_z, n)

    # Quantize
    out["x-dv-um"] = np.round(out["x-dv-um"] / res_xy) * res_xy
    out["y-ap-um"] = np.round(out["y-ap-um"] / res_xy) * res_xy
    out["z-lr-um"] = np.round(out["z-lr-um"] / res_z) * res_z

    return out


def bootstrap_v5_per_track(df, fish_id, dt=3.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    fish_df = df[df["fish_id"] == fish_id].copy()

    records = []

    for track_id, track_df in fish_df.groupby("track_id"):
        track_df = track_df.sort_values("time_point").reset_index(drop=True)

        time_points = track_df["time_point"].values
        start_pos = track_df[["x-dv-um", "y-ap-um", "z-lr-um"]].iloc[0].values

        n_steps = len(track_df) - 1

        # Single-point track — just record the position, no simulation
        if n_steps < 1:
            records.append(
                {
                    "fish_id": fish_id,
                    "track_id": track_id,
                    "time_point": time_points[0],
                    "x-dv-um": start_pos[0],
                    "y-ap-um": start_pos[1],
                    "z-lr-um": start_pos[2],
                }
            )
            continue

        # Normal case: simulate random walk
        track_step_mags = track_df["speed_um-per-min"].dropna().values * dt

        # check if track_step_mags has any nans
        track_step_mags = track_step_mags[~np.isnan(track_step_mags)]
        track_step_mags = track_step_mags[~np.isinf(track_step_mags)]
        if len(track_step_mags) == 0:
            raise ValueError(
                f"Track {track_id} has no valid step magnitudes, skipping."
            )

        # ensure we produce exactly n_steps magnitudes:
        if len(track_step_mags) >= n_steps:
            # shuffle and take first n_steps (no replacement)
            sampled_mags = np.random.permutation(track_step_mags)[:n_steps]
        else:
            # not enough mags: sample with replacement to reach n_steps
            sampled_mags = np.random.choice(track_step_mags, size=n_steps, replace=True)

        directions = random_unit_vectors(n_steps)
        steps = directions * sampled_mags[:, np.newaxis]
        positions = np.vstack([start_pos, start_pos + np.cumsum(steps, axis=0)])

        for i, pos in enumerate(positions):
            records.append(
                {
                    "fish_id": fish_id,
                    "track_id": track_id,
                    "time_point": time_points[i],
                    "x-dv-um": pos[0],
                    "y-ap-um": pos[1],
                    "z-lr-um": pos[2],
                }
            )

    return pd.DataFrame(records)


def simulate_per_track_df_concat(df_super_concat, mean_diameter_um):
    mean_diameter_px = mean_diameter_um / np.prod(NEW_PIXEL_WIDTH_UM) ** (1 / 3)
    df_sim = pd.DataFrame()

    for fid in df_super_concat.fish_id.unique():
        print(f"Processing fish_id: {fid}")
        temp_df = apply_detector_limits(
            bootstrap_v5_per_track(df_super_concat, fish_id=fid, seed=42),
            res_xy=0.5,
            res_z=1.0,
            add_noise=True,
            seed=42,
        )
        df_sim = pd.concat([df_sim, temp_df], ignore_index=True)

    # add velocity by track
    af.get_single_frame_velocity_track(df_sim, delta_t_min=3)
    # add distance_px and distance_um
    af.add_distance_um_px(df_sim, delta_t_min=3, pixel_spacing=NEW_PIXEL_WIDTH_UM)

    # add distance_px > 0.5 * mean (eqv. radius of cell)
    df_sim["distance_px>mean_radius/2"] = df_sim["distance_px"] > (
        float(mean_diameter_px) / 4
    )

    # add subtrack_id
    af.add_subtrack_info(df_sim)

    # add velocity by subtrack
    af.get_single_frame_velocity_subtrack(df_sim, delta_t_min=3)

    af.add_velocity_cosine_similarity_subtrack(df_sim)
    af.add_spatial_extent_subtrack(df_sim)
    af.add_anisotropy_index(df_sim)
    df_sim["anisotropy_index_normalized"] = np.sqrt(2) * df_sim["anisotropy_index"]
    df_sim["spatial_extent>2*mean_diameter"] = (
        df_sim["spatial_extent_um"] > 2 * mean_diameter_um
    )
    df_sim["motility"] = df_sim["spatial_extent>2*mean_diameter"]
    return df_sim


# Example usage:
# df_super_concat = pd.read_csv('path_to_your_input_file.csv')
# mean_diameter_um = 10.0  # example value
# df_simulated = simulate_per_track_df_concat(df_super_concat, mean_diameter_um)
