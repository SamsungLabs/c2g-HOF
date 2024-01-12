import os
import torch
import numpy as np
import argparse
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hof.pointnet import PointNet, PointNetParameters
from hof.model import build_HOF_Diag_rbf
from hof.visdom_saver import VisdomSaver
from seven_dof_c2g.c2g_hof_dataset import (
    C2GHOFDataset as Dataset,
)


class Cost_HOF:
    """c2g-HOF network.

    Parameters:
        args: network parameters. Please loot at config.py
        runner: Boolean for running a trained model for planning
        given_archive_path: folder path of trained model
        my_device: torch.device
    """

    def __init__(self, args, runner=False, given_archive_path=None, my_device=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.pretrained_mode = args.pt_env_name
        # training dataset folder
        datapath = "demo_data/dataset/"
        if self.pretrained_mode:
            if (runner == True) & (given_archive_path is not None):
                fn = "{}/config_args.txt".format(given_archive_path)
                print("loading from", fn)
                with open(fn, "r") as f:
                    args.__dict__ = json.load(f)
                    print("Loaded in pretrained config file")
            else:
                fn = "{}/runs/{}/config_args.txt".format(dir_path, args.pt_env_name)
                print("loading from", fn)
                with open(fn, "r") as f:
                    args.__dict__ = json.load(f)
                    print("Loaded in pretrained config file")
        else:
            self.testing_datasets = []
            self.training_datasets = []

            for file_i in range(1, 2):
                self.testing_datasets.append(
                    datapath + "7d_var_prm_c2g_cost_50_%03d.h5" % file_i
                )

            for file_i in range(1, 2):
                self.training_datasets.append(
                    datapath + "7d_var_prm_c2g_cost_50_%03d.h5" % file_i
                )

            if runner == True:
                self.testing_datasets = []
                self.training_datasets = []

        if my_device is not None:
            args.device = my_device

        self.args = args

        self.device = torch.device(self.args.device)
        self.env_name = self.args.env_name
        self.archive_path = "{}/runs/{}".format(dir_path, self.env_name)

        if not os.path.exists(self.archive_path):
            os.makedirs(self.archive_path, exist_ok=True)

        if runner == False:
            self.vs = VisdomSaver(
                self.args.visdom_port, self.env_name, self.archive_path
            )
        self.build_model()

        if not self.pretrained_mode:
            self.dump_args()
            self.dump_datasets()
            fn = self.archive_path + "/losses.txt"
            f = open(fn, "w")
            f.write("epoch\ttrain_loss\ttest_loss\n")
            f.close()
            self.start_epoch = 0
        elif runner == False:
            print("In pretrained Mode")
            self.load_dataset_files()
            print("Loaded datasets from saved dataset file")
        if runner == False:
            self.create_dataset_loaders()

    def load_dataset_files(self):
        self.training_datasets = []
        self.testing_datasets = []

        fn = self.archive_path + "/training_datasets.txt"

        with open(fn, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.training_datasets.append(line)
        fn = self.archive_path + "/testing_datasets.txt"
        with open(fn, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.testing_datasets.append(line)

    def plot_prev_losses(self):
        fn = self.archive_path + "/losses.txt"
        with open(fn, "r") as f:
            lines = f.readlines()
            # we start at the second line because the first are headers
            if self.start_epoch < (len(lines)):
                for i in range(1, self.start_epoch + 2):
                    line = lines[i].strip()
                    split_line = line.split("\t")
                    epoch = int(split_line[0])
                    train_loss = float(split_line[1])
                    test_loss = float(split_line[2])
                    self.vs.update_loss_line(epoch, train_loss, name="train")
                    self.vs.update_loss_line(epoch, test_loss, name="test")
            else:
                print("something messed up with loss file not plotting prev losses")

    def create_dataset_loaders(self):
        self.train_data_loaders = []
        self.test_data_loaders = []
        self.total_training_images = 0
        self.total_testing_images = 0

        for i in range(len(self.training_datasets)):
            self.train_data_loaders.append(
                DataLoader(
                    Dataset(h5file=self.training_datasets[i]),
                    shuffle=True,
                    batch_size=1,
                    num_workers=16,
                )
            )
            self.total_training_images += len(self.train_data_loaders[i])

        for i in range(len(self.testing_datasets)):
            self.test_data_loaders.append(
                DataLoader(
                    Dataset(h5file=self.testing_datasets[i]),
                    shuffle=True,
                    batch_size=1,
                    num_workers=16,
                )
            )
            self.total_testing_images += len(self.test_data_loaders[i])

        print("N training scenes", self.total_training_images)
        print("N testing scenes", self.total_testing_images)

    def dump_datasets(self):
        fn = self.archive_path + "/training_datasets.txt"
        f = open(fn, "w")
        for i in range(len(self.training_datasets)):
            f.write(self.training_datasets[i])
            f.write("\n")
        f.close()

        fn = self.archive_path + "/testing_datasets.txt"
        f = open(fn, "w")
        for i in range(len(self.testing_datasets)):
            f.write(self.testing_datasets[i])
            f.write("\n")
        f.close()

    def build_model(self, pretrained_hof=None):

        # build models and optimizer
        self.loss_epoch = 0
        pn_global_dim = self.args.conv_output_dim
        ws_sample_dim = self.args.ws_sample_dim
        pn_params = PointNetParameters()
        pn_params = pn_params._replace(global_feature_dim=pn_global_dim)
        self.voxel_encoder = PointNet(input_features=ws_sample_dim, params=pn_params)
        self.voxel_encoder = self.voxel_encoder.to(device=self.args.device)
        self.conv_parameters = list(self.voxel_encoder.parameters())

        self.hof_network, self.rbf_param_dim = build_HOF_Diag_rbf(
            args=self.args,
            device=self.device,
            point_dim=self.args.sample_dim * 2,
            output_dim=self.args.fc_output_dim,
        )

        self.conv_parameters.extend(list(self.hof_network.parameters()))
        self.optimizer = torch.optim.Adam(self.conv_parameters, lr=self.args.lr)

    def load_pretrained_model(self, pretrained_hof_idx):
        fn = "{}/{}_net_{}.pt".format(
            self.archive_path, self.env_name, pretrained_hof_idx
        )

        checkpoint = torch.load(fn, map_location=self.device)

        self.voxel_encoder.load_state_dict(checkpoint["primary_conv"])
        self.hof_network.load_state_dict(checkpoint["primary_fc"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch_step"]

        self.loss_epoch = self.start_epoch

    def dump_args(self):
        path = self.archive_path + "/config_args.txt"
        print("Saved arguments to:", path)
        with open(path, "w") as f:
            json.dump(self.args.__dict__, f, indent=2)

    def keep_training(self):
        epoch = input(
            "Enter epoch to continue training from. Enter 'L' for latest epoch: "
        )
        if epoch == "L":
            epoch = "LATEST"
        self.load_pretrained_model(epoch)
        self.plot_prev_losses()
        self.start_epoch += 1
        self.train()

    def run_on_test_dataset(self):
        mse = torch.nn.MSELoss()
        self.voxel_encoder.eval()
        self.hof_network.eval()

        test_loss = 0
        n_test_scenes = 0

        with torch.no_grad():
            for dl in self.test_data_loaders:
                for t, sample_dict in enumerate(dl):

                    col_pos = sample_dict["col_points"].to(device=self.device).float()
                    c2g_cost = sample_dict["c2g_cost"].to(
                        device=self.device, dtype=torch.float
                    )
                    c2g_points = sample_dict["c2g_points"].to(
                        device=self.device, dtype=torch.float
                    )

                    n_samples = 20000
                    sampled_c2g_cost, sampled_c2g_points = self.sample_c2g_points_set(
                        c2g_cost, c2g_points, n_samples
                    )
                    sampled_c2g_points = sampled_c2g_points.squeeze(0)
                    sampled_c2g_cost = sampled_c2g_cost.squeeze(0)
                    sampled_c2g_cost = sampled_c2g_cost.unsqueeze(1)

                    point_features = self.voxel_encoder(col_pos)[0]
                    theta = point_features.squeeze()
                    pred_c2g_cost = self.hof_network(sampled_c2g_points, theta)

                    loss = mse(pred_c2g_cost, sampled_c2g_cost)
                    test_loss += loss

        test_loss = torch.sqrt(test_loss / self.total_testing_images)
        print("test_loss", test_loss)
        self.vs.update_loss_line(self.loss_epoch, test_loss.item(), name="test")
        self.voxel_encoder.train()
        self.hof_network.train()
        return test_loss

    def train(self):
        """
        Training c2g-hof
        """
        # using mse loss
        mse = torch.nn.MSELoss()
        self.voxel_encoder.train()
        self.hof_network.train()
        self.loss_epoch_iters = 5

        for self.curr_epoch in range(self.start_epoch, self.args.epochs):
            num_iters = 0
            n_scenes = 0
            train_loss = 0
            list_loss = []
            train_all_loss = 0
            print("Starting Epoch", self.curr_epoch)
            for dl in self.train_data_loaders:
                for t, sample_dict in enumerate(dl):
                    num_iters = num_iters + 1

                    if np.size(np.where(np.isinf(sample_dict["c2g_cost"]))[0]) > 0:
                        raise Exception("cost-to-go should not be inf")

                    # load dataset
                    col_pos = sample_dict["col_points"].to(device=self.device).float()
                    c2g_cost = sample_dict["c2g_cost"].to(
                        device=self.device, dtype=torch.float
                    )
                    c2g_points = sample_dict["c2g_points"].to(
                        device=self.device, dtype=torch.float
                    )

                    n_samples = (
                        20000  # the number of input samples into the c2g-network
                    )
                    # sampling tuple
                    sampled_c2g_cost, sampled_c2g_points = self.sample_c2g_points_set(
                        c2g_cost, c2g_points, n_samples
                    )
                    # sampled tuple (cost, pair of configurations) from dataset
                    sampled_c2g_points = sampled_c2g_points.squeeze(0)
                    sampled_c2g_cost = sampled_c2g_cost.squeeze(0)
                    sampled_c2g_cost = sampled_c2g_cost.unsqueeze(1)

                    # c2g generating hof, 'pointnet'
                    point_features = self.voxel_encoder(col_pos)[0]
                    theta = point_features.squeeze()

                    # c2g-network with paramters from c2g generating hof
                    pred_c2g_cost = self.hof_network(sampled_c2g_points, theta)

                    loss = mse(pred_c2g_cost, sampled_c2g_cost)
                    train_loss += loss

                    if torch.isnan(loss):
                        raise Exception("loss is nan")

                    list_loss.append(loss)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if num_iters % self.loss_epoch_iters == 0:
                        self.loss_epoch = self.loss_epoch + 1

                        tmp_train_loss = train_loss
                        train_loss = torch.sqrt(train_loss / self.loss_epoch_iters)
                        if torch.isnan(train_loss):
                            raise Exception("Train loss is nan")

                        print("Training Loss:", train_loss.item())
                        self.vs.update_loss_line(
                            self.loss_epoch, train_loss.item(), name="train"
                        )

                        # self.run_on_test_dataset()
                        test_loss = self.run_on_test_dataset()
                        self.dump_losses(self.loss_epoch, train_loss, test_loss)

                        if self.loss_epoch % 25 == 0:
                            save_data = {
                                "primary_conv": self.voxel_encoder.state_dict(),
                                "primary_fc": self.hof_network.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "epoch_step": self.loss_epoch,
                            }

                            torch.save(
                                save_data,
                                "{}/{}_net_{}.pt".format(
                                    self.archive_path,
                                    self.env_name,
                                    self.loss_epoch,
                                ),
                            )
                            print("saved")
                        train_loss = 0
                        list_loss = []
            save_data = {
                "primary_conv": self.voxel_encoder.state_dict(),
                "primary_fc": self.hof_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch_step": self.loss_epoch,
            }

            torch.save(
                save_data,
                "{}/{}_net_LATEST.pt".format(self.archive_path, self.env_name),
            )

        print("Done Training")

    def dump_losses(self, epoch, train_loss, test_loss):
        fn = self.archive_path + "/losses.txt"
        f = open(fn, "a")
        f.write("{}\t{}\t{}\n".format(epoch, train_loss, test_loss))
        f.close()

    @staticmethod
    def sample_c2g_points_set(c2g_cost, c2g_points, n_samples):
        # sampling [strat, goal]
        n_c2g_points = c2g_cost.shape[1]  # the number of random goal points
        n_c2g_points2 = c2g_cost.shape[2]  # the number of random start points

        rand_index1 = np.random.randint(
            n_c2g_points, size=n_samples
        )  # randomly selected indice for goal
        rand_index2 = np.random.randint(
            n_c2g_points2, size=n_samples
        )  # randomly selected indice for start
        sampled_c2g_cost = c2g_cost[:, rand_index1, rand_index2]  # cost values

        sampled_c2g_points = torch.cat(
            (c2g_points[:, rand_index2, :], c2g_points[:, rand_index1, :]), 2
        )
        sampled_c2g_points = sampled_c2g_points  # random pair configurations
        return sampled_c2g_cost, sampled_c2g_points
