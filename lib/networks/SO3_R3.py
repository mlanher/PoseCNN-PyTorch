import torch


class SO3:
    def __init__(self):
        self.R = None

    def update(self, R: torch.Tensor):
        if R.dim() != 3 or R.size(1) != 3 or R.size(2) != 3:
            raise ValueError("Input tensor should have shape (batch_size, 3, 3)")
        self.R = R

    def log_map(self):
        batch_size = self.R.size(0)
        theta = torch.acos((self.R[:, 0, 0] + self.R[:, 1, 1] + self.R[:, 2, 2] - 1) / 2)
        w = torch.zeros(batch_size, 3, device=self.R.device)
        valid = theta != 0
        w[valid, 0] = (self.R[valid, 2, 1] - self.R[valid, 1, 2]) / (2 * torch.sin(theta[valid]))
        w[valid, 1] = (self.R[valid, 0, 2] - self.R[valid, 2, 0]) / (2 * torch.sin(theta[valid]))
        w[valid, 2] = (self.R[valid, 1, 0] - self.R[valid, 0, 1]) / (2 * torch.sin(theta[valid]))
        w[valid] *= theta[valid].view(-1, 1)
        return w

    def exp_map(self, w: torch.Tensor):
        batch_size = w.size(0)
        theta = torch.norm(w, dim=1)
        w_hat = w / theta.view(-1, 1)
        w_hat[theta == 0] = 0
        W = torch.zeros(batch_size, 3, 3, device=w.device)
        W[:, 0, 1] = -w_hat[:, 2]
        W[:, 0, 2] = w_hat[:, 1]
        W[:, 1, 0] = w_hat[:, 2]
        W[:, 1, 2] = -w_hat[:, 0]
        W[:, 2, 0] = -w_hat[:, 1]
        W[:, 2, 1] = w_hat[:, 0]
        R = torch.eye(3, device=w.device).expand(batch_size, 3, 3) + \
            torch.sin(theta).view(-1, 1, 1) * W + \
            (1 - torch.cos(theta).view(-1, 1, 1)) * torch.bmm(W, W)

        so3 = SO3()
        so3.update(R)
        return so3

    def to_matrix(self):
        return self.R

    def to_quaternion(self):
        eps = 1e-6
        ensure_pos = 1 + self.R[:, 0, 0] + self.R[:, 1, 1] + self.R[:, 2, 2] + eps
        qw = torch.sqrt(ensure_pos) / 2
        qx = (self.R[:, 2, 1] - self.R[:, 1, 2]) / (4 * qw)
        qy = (self.R[:, 0, 2] - self.R[:, 2, 0]) / (4 * qw)
        qz = (self.R[:, 1, 0] - self.R[:, 0, 1]) / (4 * qw)
        quaternion = torch.stack((qw, qx, qy, qz), dim=1)
        return quaternion

    @staticmethod
    def rand(batch_size: int):
        u = torch.randn(batch_size, 3, 3)
        q, r = torch.qr(u)
        d = torch.sign(torch.diag(r))
        R = torch.mm(q, torch.diag(d))
        so3 = SO3()
        so3.update(R)
        return so3


"""
Taken and adjusted from https://github.com/TheCamusean/grasp_diffusion/blob/master/se3dif/utils/geometry_utils.py
However, this version does not need theseus, as theseus does not support the PyTorch version used in this project.
"""
class SO3_R3():
    def __init__(self, R=None, t=None):
        self.R = SO3()
        if R is not None:
            self.R.update(R)
        self.w = self.R.log_map() if R is not None else None
        if t is not None:
            self.t = t

    def random_init(self, batch=1):
        H = self.sample(batch)
        self.R.update(H[:, :3, :3])
        self.w = self.R.log_map()
        self.t = H[:, :3, -1]
        return self

    def log_map(self):
        return torch.cat((self.t, self.w), dim=-1)

    def exp_map(self, x):
        self.t = x[..., :3]
        self.w = x[..., 3:]
        self.R = SO3().exp_map(self.w)
        return self

    def to_matrix(self):
        H = torch.eye(4).unsqueeze(0).repeat(self.t.shape[0], 1, 1).to(self.t)
        H[:, :3, :3] = self.R.to_matrix()
        H[:, :3, -1] = self.t
        return H

    # The quaternion takes the [w x y z] convention
    def to_quaternion(self, pos_scalar_only=False):
        """
        Args:
            pos_scalar_only: If True, quaternion is ensured to have a positive scalar component.
        """
        q = self.R.to_quaternion()
        if pos_scalar_only:
            mask = q[:, 0] < 0
            q[mask] = -q[mask]
        return q

    def sample(self, batch=1):
        R = SO3().rand(batch)
        t = torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)
        H[:, :3, :3] = R.to_matrix()
        H[:, :3, -1] = t
        return H
