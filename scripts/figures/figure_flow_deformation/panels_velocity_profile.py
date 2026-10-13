"""Velocity profile panel for figure_flow_deformation (matplotlib).

Panel:
- velocity_profile: Average tissue flow velocity profile (14h-26h APF)
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)


def velocity_profile(shared_data, output_base):
    """Generate average tissue flow velocity profile.

    Parameters
    ----------
    shared_data : dict
        Must contain ``"results_piv_base"`` path.
    output_base : str
        Output directory path.

    Returns
    -------
    matplotlib.figure.Figure
        The velocity profile figure.
    """
    logger.info("Generating velocity profile panel...")

    full_path = shared_data["results_piv_base"]

    time_array_path = os.path.join(
        full_path, "TimeArray_40um_olap0.5_2h_0.96olap_vi.csv"
    )
    path_to_mean_velocity_profile = os.path.join(
        full_path,
        "u_trEps_PIV_40um_olap0.5_2h_0.96olap_fltredAR0.1_indiv_trajectory_commonArea.csv",
    )
    path_to_nb_animals = os.path.join(
        full_path, "u_trEps_PIV_40um_olap0.5_2h_0.96olap_vi_mean_fltredAR0.1.csv"
    )

    min_time = 14
    max_time = 26

    # Read time array
    TimeArray = pd.read_csv(time_array_path)
    TimeArray.index = [f"frame{i + 1}" for i in range(TimeArray.shape[0])]

    time_col = "TimeArray1" if "TimeArray1" in TimeArray.columns else "Var1"
    heure = TimeArray[time_col].str.extract(r"(\d+)h").astype(int) + 1
    minute = TimeArray[time_col].str.extract(r"h(\d+)").astype(int)
    x_axis = heure.squeeze() + minute.squeeze() / 60.0

    # Read profiles
    df_mean_indiv = pd.read_csv(
        path_to_mean_velocity_profile, header=[0, 1], index_col=[0]
    )
    nb_animals_df = pd.read_csv(path_to_nb_animals, index_col=[0, 1])
    nb_animals = nb_animals_df.loc[
        :, nb_animals_df.columns.str.startswith("nb_animals")
    ].max()

    velocity_profiles = df_mean_indiv["speed"].mean()
    velocity_profiles_sem = df_mean_indiv["speed"].std() / np.sqrt(nb_animals.values)

    # Align indices
    common_index = velocity_profiles.index
    x_axis = pd.Series(x_axis, index=TimeArray.index)
    x_axis = x_axis[common_index]

    # Time filtering
    time_mask = (x_axis >= min_time) & (x_axis <= max_time)
    x_axis_filtered = x_axis[time_mask]
    velocity_profiles_filtered = velocity_profiles[time_mask]
    velocity_profiles_sem_filtered = velocity_profiles_sem[time_mask]

    def format_func(value, _):
        hours = int(value)
        minutes = int(round((value - hours) * 60))
        return f"{hours}h{minutes:02d}"

    fig = plt.figure(figsize=(10, 4))
    ax = fig.gca()

    ax.plot(
        x_axis_filtered,
        velocity_profiles_filtered,
        color="tab:orange",
        linewidth=2,
        label="Velocity",
    )
    ax.fill_between(
        x_axis_filtered,
        velocity_profiles_filtered - velocity_profiles_sem_filtered,
        velocity_profiles_filtered + velocity_profiles_sem_filtered,
        color="tab:orange",
        alpha=0.3,
    )

    # Time window patches
    for start_time, color, alpha in [(18, "#E0FFFF", 0.8), (22, "#E0BBE4", 0.4)]:
        patch = Rectangle(
            (start_time, 0),
            4,
            ax.get_ylim()[1],
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(patch)

    ax.set_xlabel("Developmental time (hAPF)", fontsize=12)
    ax.set_ylabel(
        r"Average velocity: $\Vert\mathbf{u}\Vert$ " + "($\\mu$m\u00b7h$^{-1}$)",
        fontsize=12,
    )

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.set_xlim(min_time, max_time)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Legend patches
    ax.add_patch(
        Rectangle((0, 0), 1, 1, facecolor="#E0FFFF", alpha=0.2, label="18h - 22h")
    )
    ax.add_patch(
        Rectangle((0, 0), 1, 1, facecolor="#FFDAB9", alpha=0.2, label="22h - 26h")
    )

    plt.tight_layout()
    logger.info("Velocity profile panel completed")
    return fig
