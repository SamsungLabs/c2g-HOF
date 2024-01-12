import numpy as np
from robot_planner.kinematics_gen3 import forward_kinematics


class CChecker:
    """Collision Checker.

    Parameters:
        ws_vmap: collision voxel in workspace.
        np_table: collision vexel (table) in workspace.
    """

    def __init__(self, ws_vmap: np.ndarray = None, np_table: np.ndarray = None):

        # constants
        self.tree_dim = 7
        self.CONFIGSIZE = np.array(
            [360, 257.8, 360, 259.6, 360, 240.6, 360]
        )  # +- configsize

        self.np_scene = ws_vmap
        self.np_table = np_table
        self.num_colchk = 0

    def collision_check_steps(self, node1: np.ndarray, node2: np.ndarray) -> bool:
        """Collision Checking between two nodes.

        Parameters:
            node1: node1
            node2: node2
            output:
            True/False of collision
        """
        step_num = int(self.dist(node2, node1) / self.EPSILON)
        curr_node = node1
        theta = (node2 - node1) / self.dist(node2, node1)
        for jpx in range(step_num):
            curr_node = node1 + (jpx + 1) * self.EPSILON * theta
            tf_obs, _ = self.collision_check(curr_node)
            if tf_obs:
                return True

        return False

    def collision_check(self, node: np.ndarray):
        """Collision Checking of the input node.

        Parameters:
            node: node
            output:
            True/False of collision, minimum distance to collision(not implemented)
        """

        self.num_colchk = self.num_colchk + 1
        DEC2RAD = np.pi / 180

        tf_obs = False

        curr_ang = node * DEC2RAD

        pts = forward_kinematics(curr_ang, alpha=1.0)

        for i in range(np.size(node, 0)):
            if (node[i] < -1 * self.CONFIGSIZE[i] / 2.0) | (
                node[i] > 1 * self.CONFIGSIZE[i] / 2.0
            ):
                tf_obs = True
                min_dis = 0
                return tf_obs, min_dis

        for i in range(np.size(pts, 0)):

            if i == 0:
                chk_points = np.linspace(np.array([0, 0, 0]), pts[i, 0:3], num=10)
            else:
                chka = np.linspace(pts[i - 1, 0:3], pts[i, 0:3], num=10)
                chk_points = np.concatenate((chk_points, chka), axis=0)

        max_len = 1187.2 / 1000
        chk_voxels = (chk_points + max_len) * 64 / (2 * max_len)
        chk_voxels = chk_voxels.astype(int)
        chk_voxels[chk_voxels > 63] = 63
        label0 = np.any(
            self.np_scene[chk_voxels[:, 0], chk_voxels[:, 1], chk_voxels[:, 2]] > 0.5
        )
        label1 = np.any(
            self.np_table[chk_voxels[3:, 0], chk_voxels[3:, 1], chk_voxels[3:, 2]] > 0.5
        )
        tf_obs = label0 | label1

        # need to compute min_dis well, but we skip it now for speed-up
        min_dis = 1.0

        return tf_obs, min_dis
