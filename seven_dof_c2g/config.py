import argparse
import os


def parse_arguments(parser):
    # evironment name of training
    parser.add_argument("--env_name", type=str, default="dijkstra_neg_30")
    parser.add_argument("--epochs", type=int, default=4000)  # epoch
    parser.add_argument("--device", default="cuda:0")  # device
    parser.add_argument("--visdom_port", type=int, default=8000)  # visdom port
    # pretrained environment name. automatically loads it from the runs folder
    parser.add_argument("--pt_env_name", type=str)
    parser.add_argument("--test", action="store_true", default=False)  # test or not
    # restart training from the last trained model
    parser.add_argument("--keep_training", action="store_true", default=False)
    # c2g generating HOF model
    parser.add_argument("--cnn_model_type", type=str, default="pointnet")
    # the output size of c2g generating HOF
    parser.add_argument("--conv_output_dim", type=int, default=256)
    # the dimension size of configuration
    parser.add_argument("--sample_dim", type=int, default=7)
    # workspace dimension szie
    parser.add_argument("--ws_sample_dim", type=int, default=3)
    # final output dim. this is the cost value we're estimating
    parser.add_argument("--fc_output_dim", type=int, default=1)
    # hidden neurons size
    parser.add_argument("--width", type=int, default=64)
    # learning rate
    parser.add_argument("--lr", type=float, default=0.0001)

    # for RBF Net:
    # the number of radial basis
    parser.add_argument("--rbf", type=int, default=256)
    # model type
    parser.add_argument("--rbf_type", type=str, default="diagonal")  #'full', 'diagonal'

    return parser
