import torch
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
from robot_planner.collision_checker import CChecker
import time
from scipy import spatial


def network_planning_7D(
    hof,
    args,
    col_points: np.ndarray,
    cc_checker,
    device="cpu",
    start_pos: np.ndarray = None,
    goal_pos: np.ndarray = None,
):

    """Loading the trained c2g-HOF model

    Parameters:
        hof: c2g-HOF network
        args: parameters of network
        col_points: collision points in workspace
        cc_checker: collision checker class
        device: torch.device
        start_pos: start configuration
        goal_pos:goal configuration
    """

    if (start_pos is None) or (goal_pos is None):
        raise Exception("Need start and goal configurations.")

    hof.voxel_encoder.eval()
    hof.hof_network.eval()
    print("device:", device)
    num_colchk = 0
    total_col_time = 0

    with torch.no_grad():
        col_points = np.expand_dims(col_points, axis=0)
        col_pos = torch.Tensor(col_points).to(device=device).float()

        # Network planning
        tic_time = time.time()
        waypoints = []
        goal_config = torch.Tensor(goal_pos).to(device=device).float()
        waypoints.append(start_pos)
        point_features = hof.voxel_encoder(col_pos)[0]
        theta = point_features.squeeze()

        i = 0
        at_goal = False
        enc_time = time.time() - tic_time
        curr_cost = np.inf
        curr_pos = torch.Tensor(start_pos).to(device=device)

        np_curr_pos = start_pos
        bool_local_stuck = False
        step_size = 15.0
        rand_range = 60
        num_sample_grad = 30

        while i < 200 and (at_goal is False):

            candi_pos = np_curr_pos + rand_range * (
                np.random.rand(num_sample_grad, 7) - 0.5
            )
            candi_pos[0, :] = np_curr_pos
            candi_dis = np.abs(np.linalg.norm(candi_pos - np_curr_pos, axis=1))

            candi_mask = (candi_pos < 360).all(axis=1)
            candi_pos = candi_pos[candi_mask]
            candi_dis = candi_dis[candi_mask]

            len_candi = np.size(candi_pos, 0)
            candi_pos_np = candi_pos
            candi_pos = torch.Tensor(candi_pos).to(device=device)

            if candi_pos.shape[0] == 0:
                pdb.set_trace()

            goal_points = goal_config.repeat(candi_pos.shape[0], 1)
            sampled_c2g_points = torch.cat((candi_pos, goal_points), 1)
            pred_c2g_cost = hof.hof_network(sampled_c2g_points, theta)

            pred_cost = pred_c2g_cost
            keep_going = True

            if i > 130 and keep_going:

                candi_pos_np = candi_pos.cpu().numpy().astype(float)[:, 0:7]
                goal_cost = np.linalg.norm(candi_pos_np - goal_pos, axis=1)
                goal_cost = np.square(goal_cost)
                goal_cost = np.reshape(goal_cost, (-1, 1))
                goal_cost = torch.Tensor(goal_cost).to(device=device, dtype=torch.float)
                pred_cost = pred_cost + goal_cost

            # compute gradient
            norm_grad = (
                pred_cost.cpu().numpy()[:, 0] - pred_cost.cpu().numpy()[0, 0]
            ) / candi_dis
            norm_grad = np.nan_to_num(norm_grad)

            min_list = np.argsort(norm_grad)

            # find the next position based on gradient and no collision
            for qk in range(np.size(candi_pos, 0)):
                pk = min_list[qk]
                if pk == 0:
                    continue
                srt_t = time.time()
                is_collision1, _ = cc_checker.collision_check(
                    candi_pos_np[pk]
                )  ## FIXME
                total_col_time = total_col_time + time.time() - srt_t
                num_colchk = num_colchk + 1

                chk_pos = np_curr_pos + (
                    candi_pos_np[pk, :] - np_curr_pos
                ) * step_size / np.linalg.norm((candi_pos_np[pk, :] - np_curr_pos))
                srt_t = time.time()
                is_collision2, _ = cc_checker.collision_check(chk_pos)  ## FIXME
                total_col_time = total_col_time + time.time() - srt_t
                num_colchk = num_colchk + 1

                if is_collision1 == False and is_collision2 == False:
                    next_point = pk
                    break

                if qk == (np.size(candi_pos_np, 0) - 1):
                    return np.array(waypoints), goal_pos, start_pos, 0, 0, False

            curr_cost = pred_cost.cpu().numpy()[next_point]
            if next_point != 0:
                curr_pos = curr_pos + (
                    candi_pos[next_point, :] - curr_pos
                ) * step_size / torch.norm((candi_pos[next_point, :] - curr_pos))
                bool_local_stuck = False
            else:
                bool_local_stuck = True

            waypoints.append(curr_pos.cpu().numpy()[:7])
            np_curr_pos = curr_pos.cpu().numpy().ravel()[:7]
            np_config_goal = goal_pos

            if spatial.distance.euclidean(np_curr_pos, np_config_goal) < 30:
                candidates = np.linspace(np_curr_pos, np_config_goal, 5)
                failed = False

                for pk in range(np.size(candidates, 0) - 2):
                    srt_t = time.time()
                    is_collision1, _ = cc_checker.collision_check(
                        candidates[pk + 1, :]
                    )  # FIXME
                    total_col_time = total_col_time + time.time() - srt_t
                    num_colchk = num_colchk + 1

                    if is_collision1 == True:
                        keep_going = True
                        failed = True
                        break

                if not failed:
                    n_success = 1
                    successful_traj = True
                    keep_going = False
                    at_goal = True
                    for pk in range(np.size(candidates, 0) - 1):
                        waypoints.append(candidates[pk + 1, :])

            i += 1
        plan_time = time.time() - tic_time
        print("Planning Done!")

        return np.array(waypoints), goal_pos, start_pos, enc_time, plan_time, at_goal
