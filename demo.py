"""
Script to run c2g-hof on demo data
"""

import numpy as np
import torch
import argparse
import seven_dof_c2g.config as config
from seven_dof_c2g.c2g_hof import Cost_HOF
import pickle
from seven_dof_c2g.c2g_hof_dataset import (
    C2GHOFDataset as Dataset,
)
from robot_planner.collision_checker import CChecker
from robot_planner.network_planners import network_planning_7D as network_planning
from robot_planner.visualization import plot_traj7D as plot_traj


def load_hof(device="cpu", archive_path: chr = None, pt_env_name: chr = None):
    """Loading the trained c2g-HOF model

    Parameters:
        device: torch.device
        archive_path: folder path of trained model
        pt_env_name: pre-trained model environment's name
    """

    if (archive_path is None) or (pt_env_name is None):
        raise Exception("Need an archive path for trained model")

    # load network
    parser = argparse.ArgumentParser()
    parser = config.parse_arguments(parser)
    args = parser.parse_args()
    args.pt_env_name = pt_env_name

    hof = Cost_HOF(
        args,
        runner=True,
        given_archive_path=archive_path + pt_env_name,
        my_device=device,
    )
    hof.load_pretrained_model(2000)  # 2000 epoch model

    return hof


if __name__ == "__main__":

    """
    The planning demo requires robot kinematics and user-defined workspace.
    The robot kinematics and workspace should be same as the definition for dataset.
    In this example, we use Gen3 robot kinematics and pointclouds in worksapce.
    This example has the same format and robot the example dataset.
    Please modify kinematics and workspace freely.
    """
    ## The trained c2g-hof network
    pt_env_name = "ws_7d_prm_var_c2g_diag_rbf_0005"  # pre-trained model name
    archive_path = "seven_dof_c2g/runs/"  # archive path for the trained model

    # load the model
    hof = load_hof(
        device="cuda:0",
        archive_path="seven_dof_c2g/runs/",
        pt_env_name="ws_7d_prm_var_c2g_diag_rbf_0005",
    )

    # load demo data for planning
    demo_data_path = "demo_data/demo_plan/"
    f3 = open(demo_data_path + "example_planning.pickle", "rb")
    configs_planning = pickle.load(f3)
    f3.close()

    # Voxelization of workspace (64 X 64 X 64)
    # obstacle points in workspace
    col_points = configs_planning["col_points"]
    # obstacle dimension and location in workspace
    locations = configs_planning["obstacle_locs"]
    lengths = configs_planning["obstacle_lengths"]

    # voxel of obstacle for collision checker
    ws_vmap = configs_planning["ws_vmap"]
    # voxel of table
    np_table = configs_planning["np_table"]

    # example start and goal configuration, degrees
    start_pos = np.array(
        [
            90.0,
            20.87527941,
            -179.99984796,
            -106.42405024,
            -0.81029144,
            -52.70067031,
            90.14330839,
        ]
    )
    goal_pos = np.array(
        [
            -90.0,
            43.59339415,
            -179.99984796,
            -113.48887778,
            0.81029144,
            67.08234335,
            89.94117613,
        ]
    )
    # define a collision checker
    cc_checker = CChecker(ws_vmap=ws_vmap, np_table=np_table)

    # run planner with c2g-HOF
    waypoints, _, _, _, plan_time, _ = network_planning(
        hof=hof,
        args=hof.args,
        col_points=col_points,
        cc_checker=cc_checker,
        device=hof.args.device,
        start_pos=start_pos,
        goal_pos=goal_pos,
    )
    print("planning time:", plan_time)
    ## visualization of the generated trajectory
    plot_traj(locations, lengths, waypoints)
