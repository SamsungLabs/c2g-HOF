import os
import warnings
import h5py
import numpy as np
from torch.utils.data import Dataset

# Galen H5Dataset
class C2GHOFDataset(Dataset):
    def __init__(
        self,
        h5file,
    ):
        self.h5f = h5file
        h5f = h5py.File(self.h5f, "r")
        self.__length = len(h5f["c2g_points"])
        print(self.__length)
        if self.__length < 50:
            print(self.h5f)
        h5f.close()

    def __len__(self):
        return self.__length

    def __get_file_contents(self, idx):
        with h5py.File(self.h5f, "r") as h5:

            ob_locs = h5["dims"][str(idx)][:]
            ob_length = h5["lengths"][str(idx)][:]
            c2g_cost = h5["c2g_cost"][str(idx)][:].astype(float)
            c2g_cost = c2g_cost
            c2g_points = h5["c2g_points"][str(idx)][:].astype(float)  # configurations
            col_points = h5["col_ws_pts"][str(idx)][:].astype(
                float
            )  # collision points in workspace
            col_points = col_points[np.random.choice(np.size(col_points, 0), 10000), :]
            sample_dict = {
                "col_points": col_points,
                "obstacle_locs": ob_locs,
                "obstacle_lengths": ob_length,
                "c2g_cost": c2g_cost,
                "c2g_points": c2g_points,
            }

        return sample_dict

    def __getitem__(self, idx):
        return self.__get_file_contents(idx)
