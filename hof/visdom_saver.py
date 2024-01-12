import visdom
import matplotlib.pyplot as plt
import numpy as np
import pdb


class VisdomSaver:
    """Visdom saver.

    Parameters:
        port: port number.
        env_name: training environment name.
        archive_path: folder path.
    """

    def __init__(self, port: int, env_name: str, archive_path: str):
        self.loss_plot = None
        self.acc_plot = None
        self.env_name = env_name

        self.archive_path = archive_path

        self.viz = visdom.Visdom(
            port=port, log_to_filename="{}/visdom.json".format(self.archive_path)
        )

    def update_acc_line(self, x, acc, env_name=None, name=None):
        self.acc_plot = self.viz.line(
            [acc],
            [x],
            env=env_name if env_name is not None else self.env_name,
            win=self.acc_plot,
            name=name,
            update="append" if self.acc_plot else None,
        )

    def update_loss_line(self, x, loss, env_name=None, name=None):
        self.loss_plot = self.viz.line(
            [loss],
            [x],
            env=env_name if env_name is not None else self.env_name,
            win=self.loss_plot,
            name=name,
            update="append" if self.loss_plot else None,
        )

    def display_image(self, image, title=None, env_name=None):
        self.viz.image(
            image,
            env=env_name if env_name is not None else self.env_name,
            opts=dict(webgl=True, title=title),
        )
