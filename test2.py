import open3d as o3d
import math
import numpy as np

import torch
import torch.nn as nn
from src.utils import spec_gaussian_filter, fftfreqs, img, grid_interp, point_rasterize
from skimage import measure
import torch.fft


in_pc_file = 'data/demo/wheel.ply'
out_mesh_file = 'out.ply'


class DPSR(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        """
        :param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR, self).__init__()
        self.res = res
        self.sig = sig
        self.dim = len(res)
        self.denom = np.prod(res)
        G = spec_gaussian_filter(res=res, sig=sig).float()
        # self.G.requires_grad = False # True, if we also make sig a learnable parameter
        self.omega = fftfreqs(res, dtype=torch.float32)
        self.scale = scale
        self.shift = shift
        self.register_buffer("G", G)

    def forward(self, V, N):
        """
        :param V: (batch, nv, 2 or 3) tensor for point cloud coordinates
        :param N: (batch, nv, 2 or 3) tensor for point normals
        :return phi: (batch, res, res, ...) tensor of output indicator function field
        """
        assert (V.shape == N.shape)  # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]

        ras_s = torch.fft.rfftn(ras_p, dim=(2, 3, 4))
        ras_s = ras_s.permute(*tuple([0] + list(range(2, self.dim + 1)) + [self.dim + 1, 1]))
        # N_ = ras_s[..., None] * self.G  # [b, dim0, dim1, dim2/2+1, n_dim, 1]
        N_ = ras_s[..., None]  # [b, dim0, dim1, dim2/2+1, n_dim, 1]

        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1)  # [dim0, dim1, dim2/2+1, n_dim, 1]
        omega *= 2 * np.pi  # normalize frequencies
        omega = omega.to(V.device)

        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)

        Lap = -torch.sum(omega ** 2, -2)  # [dim0, dim1, dim2/2+1, 1]
        Phi = DivN / (Lap + 1e-6)  # [b, dim0, dim1, dim2/2+1, 2]
        Phi = Phi.permute(*tuple([list(range(1, self.dim + 2)) + [0]]))  # [dim0, dim1, dim2/2+1, 2, b]
        Phi[tuple([0] * self.dim)] = 0
        Phi = Phi.permute(*tuple([[self.dim + 1] + list(range(self.dim + 1))]))  # [b, dim0, dim1, dim2/2+1, 2]

        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1, 2, 3))

        if self.shift or self.scale:
            # ensure values at points are zero
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1)  # [b, nv]
            if self.shift:  # offset points to have mean of 0
                offset = torch.mean(fv, dim=-1)  # [b,]
                phi -= offset.view(*tuple([-1] + [1] * self.dim))

            phi = phi.permute(*tuple([list(range(1, self.dim + 1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))

            if self.scale:
                phi = -phi / torch.abs(fv0.view(*tuple([-1] + [1] * self.dim))) * 0.5
        return phi


class DPSR2(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        """
        :param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR2, self).__init__()
        self.res = res
        #self.sig = sig
        self.dim = len(res)
        self.denom = np.prod(res)
        #G = spec_gaussian_filter(res=res, sig=sig).float()
        # self.G.requires_grad = False # True, if we also make sig a learnable parameter
        self.omega = fftfreqs(res, dtype=torch.float32)
        self.scale = scale
        self.shift = shift
        #self.register_buffer("G", G)

    def forward(self, V, N):
        """
        :param V: (batch, nv, 2 or 3) tensor for point cloud coordinates
        :param N: (batch, nv, 2 or 3) tensor for point normals
        :return phi: (batch, res, res, ...) tensor of output indicator function field
        """
        assert(V.shape == N.shape) # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]
        # ras_p.shape: [1, 3, 64, 64, 64]

        ras_s = torch.fft.rfftn(ras_p, dim=(2,3,4))
        # ras_s.shape: [1, 3, 64, 64, 33]

        # [0, 2, 3, 4, 1] when dim=3.
        ras_s = ras_s.permute(*tuple([0]+list(range(2, self.dim+1))+[self.dim+1, 1]))
        # ras_s.shape: [1, 64, 64, 33, 3]

        # G.shape: [64, 64, 33, 1, 1]

        '''
        #N_ = ras_s[..., None] * self.G # [b, dim0, dim1, dim2/2+1, n_dim, 1]
        N_ = ras_s[..., None]
        # N_.shape: [1, 64, 64, 33, 3, 1]
        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1) # [dim0, dim1, dim2/2+1, n_dim, 1]
        # omega.shape: [64, 64, 33, 3, 1]
        '''

        N_ = ras_s
        # N_.shape: [1, 64, 64, 33, 3]
        omega = fftfreqs(self.res, dtype=torch.float32) # [dim0, dim1, dim2/2+1, n_dim]
        # omega.shape: [64, 64, 33, 3]


        #### Original implementation - Continuous FT version ####
        ################################
        omega_w = 2 * np.pi * omega.unsqueeze(-1)  # normalize frequencies

        DivN_c = -torch.sum(-img(torch.view_as_real(N_)) * omega_w, dim=-2)
        # DivN_c.shape: [1, 64, 64, 33, 2]

        Lap_c = -torch.sum(omega_w**2, -2) # [dim0, dim1, dim2/2+1, 1]
        # Lap_c.shape: [64, 64, 33, 1]
        ################################



        #### Discrete FT version ####
        ################################
        # NOTE: Assume that the volume size along each axis is 1.
        self.one_over_intv = torch.from_numpy(np.array([self.res]))
        # self.one_over_intv.shape = [3]

        k_over_N = [None] * self.dim
        for i in range(self.dim):
            if (i + 1) == self.dim:
                # [0, 1, ..., n/2]
                k_over_N[i] = np.concatenate([
                    np.arange(0, math.floor((self.res[i])/2) + 1)]) / self.res[i]
            else:
                # [0, 1, ..., n/2-1, -n/2, ..., -1] if n is even.
                # [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] if n is odd.
                k_over_N[i] = np.concatenate([
                    np.arange(0, math.floor((self.res[i] + 1)/2)), \
                    np.arange(-math.floor(self.res[i] / 2), 0)]) / self.res[i]

        self.k_over_N_grid = torch.from_numpy(np.array(np.meshgrid(*k_over_N)))
        # self.k_over_N_grid.shape = [3, 64, 64, 33]

        self.k_over_N_grid = self.k_over_N_grid.permute(*tuple(list(range(1, self.dim+1))+[0]))
        # self.k_over_N_grid.shape = [64, 64, 33, 3]

        #### Divergence ####
        if True:
            Div_w = self.one_over_intv
            Div_w = Div_w.reshape((1,)*self.dim+(self.dim,))
            # Div_w.shape: [1, 1, 1, 3]

            # Div_Op_R = (Div_w * (torch.cos(2.0 * math.pi * self.k_over_N_grid) - 1))
            Div_Op_R = torch.zeros_like(self.k_over_N_grid)
            Div_Op_I = (Div_w * torch.sin(2.0 * math.pi * self.k_over_N_grid))
            # Div_R.shape: [64, 64, 33, 3]
            # Div_I.shape: [64, 64, 33, 3]
        else:
            #### TEST - Continuous FT version ####
            Div_Op_R = torch.zeros_like(omega)
            Div_Op_I = 2 * np.pi * omega
            # Div_R.shape: [64, 64, 33, 3]
            # Div_I.shape: [64, 64, 33, 3]

        Div_Op = torch.view_as_complex(torch.stack([Div_Op_R, Div_Op_I], dim=-1))
        # Div_Op.shape: [64, 64, 33, 3]

        # N_.shape: [1, 64, 64, 33, 3]
        DivN_d = torch.view_as_real(torch.sum(Div_Op * N_, dim=-1))
        # DivN.shape: [1, 64, 64, 33, 2]


        #### Laplacian ####
        Lap_w = 2 * torch.square(self.one_over_intv)
        Lap_w = Lap_w.reshape((1,)*self.dim+(self.dim,))
        # Lap_w.shape: [1, 1, 1, 3]

        Lap_d = torch.sum(Lap_w * (torch.cos(2.0 * math.pi * self.k_over_N_grid) - 1), dim=-1)
        Lap_d = Lap_d.unsqueeze(-1)
        # Lap.shape: [64, 64, 33, 1]
        ################################



        #### Choose one of the followings:
        Phi = DivN_d / (Lap_d+1e-6) # [b, dim0, dim1, dim2/2+1, 2]
        # Phi = DivN_c / (Lap_c+1e-6) # [b, dim0, dim1, dim2/2+1, 2]
        # Phi = DivN_c / (Lap_d+1e-6) # [b, dim0, dim1, dim2/2+1, 2]
        # Phi: [1, 64, 64, 33, 2]

        # [1, 2, 3, 4, 0] with dim=3.
        Phi = Phi.permute(*tuple([list(range(1,self.dim+2)) + [0]]))  # [dim0, dim1, dim2/2+1, 2, b]
        # Phi.shape: [64, 64, 33, 2, 1]

        # Already zero...
        Phi[tuple([0] * self.dim)] = 0

        # [4, 0, 1, 2, 3] with dim=3
        Phi = Phi.permute(*tuple([[self.dim+1] + list(range(self.dim+1))]))  # [b, dim0, dim1, dim2/2+1, 2]
        # Phi.shape: [1, 64, 64, 33, 2]

        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1,2,3))
        # phi.shape: [1, 64, 64, 64]

        if self.shift or self.scale:
            # ensure values at points are zero
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1) # [b, nv]
            if self.shift: # offset points to have mean of 0
                offset = torch.mean(fv, dim=-1)  # [b,]
                phi -= offset.view(*tuple([-1] + [1] * self.dim))

            phi = phi.permute(*tuple([list(range(1,self.dim+1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))

            if self.scale:
                phi = -phi / torch.abs(fv0.view(*tuple([-1]+[1] * self.dim))) *0.5

        return phi


def implicit_to_mesh(target, out_mesh_file):
    target = torch.tanh(target)
    s = target.shape[-1] # size of psr_grid
    psr_grid_numpy = target.squeeze().detach().cpu().numpy()
    verts, faces, _, _ = measure.marching_cubes(psr_grid_numpy)
    verts = verts / s * 2. - 1 # [-1, 1]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(out_mesh_file, mesh)


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(in_pc_file)
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)
    P = np.expand_dims(P, 0)
    N = np.expand_dims(N, 0)
    print(np.shape(P))
    print(np.shape(N))

    # Normalize the input point cloud to the range of [0, 1].
    P_min, P_max = np.amin(P), np.amax(P)
    P_min -= 0.1 * (P_max - P_min)
    P_max += 0.1 * (P_max - P_min)
    P = (P - P_min) / (P_max - P_min)

    P = torch.from_numpy(P)
    N = torch.from_numpy(N)

    res = 128
    dpsr = DPSR2(res=(res, res, res), scale=False, shift=False)
    phi = dpsr(P, N)
    implicit_to_mesh(phi, out_mesh_file)
