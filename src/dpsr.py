import torch
import torch.nn as nn
from src.utils import spec_gaussian_filter, fftfreqs, img, grid_interp, point_rasterize
import numpy as np
import torch.fft


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
        assert(V.shape == N.shape) # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]
        
        ras_s = torch.fft.rfftn(ras_p, dim=(2,3,4))
        ras_s = ras_s.permute(*tuple([0]+list(range(2, self.dim+1))+[self.dim+1, 1]))
        N_ = ras_s[..., None] * self.G # [b, dim0, dim1, dim2/2+1, n_dim, 1]

        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1) # [dim0, dim1, dim2/2+1, n_dim, 1]
        omega *= 2 * np.pi  # normalize frequencies
        omega = omega.to(V.device)
        
        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)
        
        Lap = -torch.sum(omega**2, -2) # [dim0, dim1, dim2/2+1, 1]
        Phi = DivN / (Lap+1e-6) # [b, dim0, dim1, dim2/2+1, 2]  
        Phi = Phi.permute(*tuple([list(range(1,self.dim+2)) + [0]]))  # [dim0, dim1, dim2/2+1, 2, b] 
        Phi[tuple([0] * self.dim)] = 0
        Phi = Phi.permute(*tuple([[self.dim+1] + list(range(self.dim+1))]))  # [b, dim0, dim1, dim2/2+1, 2]
        
        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1,2,3))
        
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


def freq_tensor(shape):
    dim = len(shape)
    w_shape = shape[:-1] + (shape[-1] // 2 + 1,) + (dim,)
    w = torch.cartesian_prod(*[torch.fft.fftfreq(shape[idx], dtype=torch.double) if idx != dim - 1 else
                               torch.fft.rfftfreq(shape[idx], dtype=torch.double)
                               for idx in range(dim)]).reshape(w_shape)
    return w


def div_op(shape, spacings):
    import math
    sin_w = torch.sin(2 * math.pi * freq_tensor(shape)) / spacings
    return torch.complex(torch.zeros_like(sin_w), sin_w)


def ilap_op(shape, spacings):
    import math
    cos_w = torch.cos(2 * math.pi * freq_tensor(shape))
    ilap = torch.reciprocal(torch.sum(2 * (cos_w - 1) / (spacings ** 2), dim=-1))
    ilap[(0,) * len(ilap.shape)] = 0
    return ilap


def scale_and_shift(phi_n, pc_r):
    phi_n -= torch.mean(grid_interp(phi_n.unsqueeze(-1), pc_r[:, 0, :, :]).squeeze(-1), dim=1)
    dim = pc_r.shape[-1]
    idx = (slice(None),) + (0,) * dim
    phi_n = phi_n * 0.5 / torch.abs(phi_n[idx])
    return phi_n


class DFTPSR(nn.Module):
    def __init__(self, shape, sig, device="cuda"):
        super(DFTPSR, self).__init__()
        self.shape = shape
        self.fft_dims = tuple(range(1, len(self.shape) + 1))
        spacings = tuple([1 / res for res in shape])
        spacings = torch.tensor(spacings, dtype=torch.double)
        self.div = div_op(self.shape, spacings).to(device)
        self.ilap = ilap_op(self.shape, spacings).to(device)

    def rasterize(self, pc_r):
        sap_raster = point_rasterize(pc_r[:, 0, ...], pc_r[:, 1, ...], self.shape)
        return torch.moveaxis(sap_raster, 1, -1)

    def forward(self, v_r, n_r):
        pc_r = torch.stack([v_r, n_r], dim=1)
        v_n = self.rasterize(pc_r)
        v_k = torch.fft.rfftn(v_n, dim=self.fft_dims)
        f_k = torch.sum(self.div * v_k, -1)
        phi_n = torch.fft.irfftn(self.ilap * f_k, dim=self.fft_dims)
        phi_n = scale_and_shift(phi_n, pc_r)
        return phi_n
