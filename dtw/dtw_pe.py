import numpy as np

import torch
import torch.nn as nn

from dtaidistance import dtw, dtw_ndim


class DTWPositionalEncoding(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.original_pos_enc = None
        self.source_mean_pe = None
        self.device = device
        self.dtw_max_step = None
        self.dtw_window = None
        self.dtw_psi = None

    def forward(self, x):
        assert self.source_mean_pe is not None, "Source mean PE not set"

        # Get PE values
        x = self.original_pos_enc(x)

        pes = x.detach().cpu().numpy()
        pes = np.double(pes)
        new_pe = np.zeros(pes.shape)

        # Compute DTW paths
        if isinstance(self.source_mean_pe, list):
            # if multiple shapes, compute DTW for each shape
            # select the path corresponding to the lowest distance

            distances = np.array([[dtw_ndim.distance_fast(mean_pe, pe, psi=self.dtw_psi, window=self.dtw_window, max_step=self.dtw_max_step)
                                   for mean_pe in self.source_mean_pe] for pe in pes])
            selected_idx = np.argmin(distances, axis=1)

            paths = []
            for i, pe in enumerate(pes):
                _, ps = dtw_ndim.warping_paths(
                    self.source_mean_pe[selected_idx[i]], pe, psi=self.dtw_psi, 
                    window=self.dtw_window, max_step=self.dtw_max_step, use_c=True)
                path = dtw.best_path(ps)
                paths.append(path)

            for b, path in enumerate(paths):
                for i, j in path:
                    new_pe[b][j] = self.source_mean_pe[selected_idx[b]][i]

        else:
            paths = []
            for i, pe in enumerate(pes):
                _, ps = dtw_ndim.warping_paths_fast(
                    self.source_mean_pe, pe, psi=self.dtw_psi, window=self.dtw_window, max_step=self.dtw_max_step)
                path = dtw.best_path(ps)
                paths.append(path)

            for b, path in enumerate(paths):
                for i, j in path:
                    new_pe[b][j] = self.source_mean_pe[i]

        return torch.Tensor(new_pe).to(self.device)

    def set_source_pes(self, source_pes):
        self.source_mean_pe = source_pes
