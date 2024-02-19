import json
import numpy as np
import torch
from .models import *


class Pose3dLifter:
    def __init__(self, cfg_file, weight, device="cpu", img_size=(224, 224), num_kps=17):
        self.device = device
        self.num_kps = num_kps
        self.height, self.width = img_size
        with open(cfg_file, 'r') as f:
            args = json.load(f)
        if args.arch == 'gcn':
            adj = adj_mx_from_skeleton(dataset.skeleton())
            self.pose3d_model = SemGCN(adj, 128, num_layers=args.stages, p_dropout=args.dropout, nodes_group=None).to(device)

        elif args.args.arch == 'stgcn':
            self.pose3d_model = WrapSTGCN(p_dropout=args.dropout).to(device)

        elif args.args.arch == 'mlp':
            self.pose3d_model = LinearModel(num_kps * 2, (num_kps - 1) * 3, num_stage=args.stages,
                                            linear_size=args.linear_size)

        elif args.args.arch == 'videopose':
            filter_widths = [1]
            for stage_id in range(args.stages):
                filter_widths.append(1)  # filter_widths = [1, 1, 1, 1, 1]
            self.pose3d_model = TemporalModelOptimized1f(16, 2, 15, filter_widths=filter_widths, causal=False,
                                                         dropout=0.25, channels=1024)
        else:
            assert False, 'posenet_name invalid'

        torch.load(weight, map_location=device)

    def process(self, kps):
        norm_kps = self.normalize_screen_coordinates(kps[..., :2], w=self.width, h=self.height)
        outputs_3d = self.pose3d_model(norm_kps.view(self.num_kps, -1)).view(self.num_kps, -1, 3).cpu()
        outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]  # the output is relative to the 0 joint
        return outputs_3d

    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def visualize(self, channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
                   gt=False, pred=False):  # blue, orange
        #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
        vals = np.reshape(channels, (16, -1))

        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

        # Make connection matrix
        for i in np.arange(len(I)):
            x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
            if gt:
                ax.plot(x, z, -y, lw=2, c='k')
            #        ax.plot(x,y, z,  lw=2, c='k')

            elif pred:
                ax.plot(x, z, -y, lw=2, c='r')
            #        ax.plot(x,y, z,  lw=2, c='r')

            else:
                #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
                ax.plot(x, z, -y, lw=2, c=lcolor if LR[i] else rcolor)

        RADIUS = 1  # space around the subject
        xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
        ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
        ax.set_ylim3d([-RADIUS + zroot, RADIUS + zroot])
        ax.set_zlim3d([-RADIUS - yroot, RADIUS - yroot])

        if add_labels:
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_zlabel("-y")
        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)




