import numpy as np
import torch
from torch.utils.data import DataLoader

# from nerf.provider
def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (right) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (left) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def circle_poses(device, radius=1.25, theta=60.0, phi=0.0, angle_overhead=30.0, angle_front=60.0):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)
    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return dirs, thetas.item(), phis.item(), radius

class MultiviewDataset:
    def __init__(self, cfg, size, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, tests

        self.phis = [(index / size) * 360 for index in range(size)]
        self.thetas = [90 for _ in range(size)]

        # Alternate lists
        if size % 2 ==0:
            alternate_lists = lambda l: [l[0]] + [i for j in zip(l[1:size // 2], l[-1:size // 2:-1]) for i in j] + [
                l[size // 2]]
        else:
            alternate_lists = lambda l: [l[0]] + [i for j in zip(l[1:size // 2 + 1], l[-1:size // 2:-1]) for i in j]
        # Alternate views
        self.phis = alternate_lists(self.phis)
        self.thetas = alternate_lists(self.thetas)

        # for phi, theta in self.cfg.views_before:
        #     self.phis = [phi] + self.phis
        #     self.thetas = [theta] + self.thetas
        # for phi, theta in self.cfg.views_after:
        #     self.phis = self.phis + [phi]
        #     self.thetas = self.thetas + [theta]
        #     # self.phis = [0, 0] + self.phis
        #     # self.thetas = [20, 160] + self.thetas

        self.size = len(self.phis)

    def collate(self, index):

        # B = len(index)  # always 1

        # phi = (index[0] / self.size) * 360
        phi = self.phis[index[0]]
        theta = self.thetas[index[0]]
        radius = self.cfg.radius
        dirs, thetas, phis, radius = circle_poses(self.device, radius=radius, theta=theta,
                                                  phi=phi,
                                                  angle_overhead=self.cfg.angle_overhead,
                                                  angle_front=self.cfg.angle_front)

        data = {
            'dir': dirs,
            'theta': thetas,
            'phi': phis,
            'radius': radius
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader
