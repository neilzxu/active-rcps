import numpy as np
import torch


class FalsePositiveRho(object):

    def __init__(self):
        super().__init__()

    def __call__(self, PXY, betas):
        import numpy as np
        (P, _), Y = PXY
        return (1 - Y) * (betas <= P)

    def sample(self):
        import torch

        p = torch.rand(1)
        y = torch.bernoulli(p)
        return p, y


class TopBetaCoverage(object):

    def __init__(self, w=None, torch_w=None):
        import torch

        import numpy as np

        super().__init__()
        if torch_w is None:
            self.w = w if w is not None else np.ones(1000)
            self.torch_w = torch.Tensor(self.w)
        else:
            self.torch_w = torch_w
            self.w = torch_w.detach().numpy()

    def _coverage_mask(self, idx_arr, top_idx):
        """Get mark array that indicates the labeled that are covered :param
        w_P: n x classes tensor that's label cost times predicted prob.

        for label
        :param idx_arr: n x classes tensor that class indices sorted by
            highest to lowest predicted prob.
        :param top_idx: n len tensor of number of indices covered under
            a choice of beta
        """
        import torch
        mask = torch.zeros(idx_arr.shape).bool()
        row_indices = torch.cat([
            torch.full((top_idx[i].item() + 1, ), i)
            for i in range(len(top_idx))
        ])
        col_indices = torch.cat(
            [idx_arr[i, :(top_idx[i] + 1)] for i in range(len(top_idx))])
        mask[row_indices, col_indices] = True
        return mask

    def beta_coverage_set(self, P, betas):
        import torch

        w_P = P * self.torch_w.reshape(1, -1)
        sort_arr, idx_arr = torch.sort(w_P,
                                       descending=True,
                                       dim=-1,
                                       stable=True)
        sums_arr = torch.cumsum(sort_arr, dim=-1)
        top_idx = torch.argmax((sums_arr >= betas).int(), dim=-1)

        # if all sums are below beta, set to last possible index (i.e., include all labels)
        sum_small_idxs = torch.all(sums_arr < betas, dim=-1)
        top_idx[sum_small_idxs] = self.torch_w.shape[0] - 1

        return self._coverage_mask(idx_arr, top_idx)

    def _compute_loss(self, P, Y, betas):
        import torch
        w_P = P * self.torch_w.reshape(1, -1)
        sort_arr, idx_arr = torch.sort(w_P,
                                       descending=True,
                                       dim=-1,
                                       stable=True)
        sums_arr = torch.cumsum(sort_arr, dim=-1)
        top_idx = torch.argmax((sums_arr >= betas).int(), dim=-1)

        # if all sums are below beta, set to last possible index (i.e., include all labels)
        sum_small_idxs = torch.all(sums_arr < betas, dim=-1)
        top_idx[sum_small_idxs] = self.torch_w.shape[0] - 1
        top_idx = top_idx.reshape(-1, 1)

        label_top_idx = torch.argmax((idx_arr == Y.reshape(-1, 1)).int(),
                                     dim=-1).reshape(-1, 1)
        miscover_arr = (label_top_idx > top_idx).float()
        # assert torch.all(Y <= 1000), Y
        return miscover_arr.reshape(-1) * self.torch_w[Y].reshape(-1)

    def data_covsize_curve(self, P):
        w_P = P * self.torch_w.reshape(1, -1)
        sorted_probs = torch.sort(w_P, descending=True, axis=1).values.cumsum(
            axis=1)[:, :(-1)]  # drop last column since probs add up to 1
        sorted_min_betas = torch.sort(sorted_probs.reshape(-1)).values
        total_items = len(sorted_min_betas)

        # we're doing the tail bound integration version of expectation
        def avg_covsize(beta):
            idx = torch.searchsorted(
                sorted_min_betas, beta,
                right=True)  # length of things less than or equal to beta
            return 1 + (idx / total_items)

        return avg_covsize

    def data_risk_curve(self, P, Y):
        w_P = P * self.torch_w.reshape(1, -1)
        sorted_obj = torch.sort(w_P, desceding=True, axis=1)
        sorted_prob_idxs = sorted_obj.indices
        sorted_probs = sorted_obj.values.cumsum(axis=1)
        min_betas = torch.sort(
            sorted_probs[sorted_prob_idxs == Y]
        )  # if beta is greater than this, then this data point gets covered
        total_items = len(min_betas)

        def avg_risk(beta):
            idx = torch.searchsorted(
                min_betas, beta,
                right=True)  # length of things less than or equal to beta
            return idx / total_items

        return avg_risk

    def __call__(self, PXY, betas, is_torch=False, betas_is_torch=False):
        """Either beta can be single or multiple rows (i.e., single beta)"""

        (P, _), Y = PXY

        if isinstance(betas, np.ndarray):
            betas = torch.tensor(betas)
        elif not isinstance(betas, torch.Tensor):
            betas = torch.as_tensor(betas)
        betas = betas.reshape(-1, 1)
        # betas = (betas if betas_is_torch else torch.Tensor([betas])).reshape(
        #     -1, 1)

        # Either a batch of P or a batch of betas, but not both
        assert len(P.shape) == 1 or (P.shape[0] == 1) or betas.shape[0] == 1, (
            P.shape, betas.shape)
        if len(P.shape) == 1:
            P = P.reshape(1, -1)

        if is_torch:
            res = self._compute_loss(P, Y, betas)
        else:
            res = self._compute_loss(torch.Tensor(P),
                                     torch.Tensor([Y]).reshape(-1).int(),
                                     betas).numpy()

        if betas.shape[0] != 1:
            return res.reshape(-1)
        else:
            return res
        # return (np.argmax(P, axis=-1) == Y)

    def sample(self):
        import torch

        p = torch.rand(1)
        y = torch.bernoulli(p)
        return p, y


class WeightedCoverage(object):

    def __init__(self, w=None):
        import numpy as np
        import torch

        super().__init__()
        self.w = w if w is not None else np.ones(1000)
        self.torch_w = torch.Tensor(self.w)

    def beta_coverage_set(self, P, betas):
        threshold = 1 - betas
        w_P = P * self.torch_w.reshape(1, -1)
        return w_P >= threshold

    def __call__(self, PXY, betas, is_torch=False):
        import numpy as np
        (P, _), Y = PXY
        threshold = 1 - betas
        if is_torch:
            return (1. - self.beta_coverage_set(
                P, betas)[np.arange(P.shape[0]), Y].float()) * self.torch_w[Y]
        else:
            return (1. - self.beta_coverage_set(
                P, betas)[np.arange(P.shape[0]), Y]) * self.w[Y]

    def sample(self):
        import torch

        p = torch.rand(1)
        y = torch.bernoulli(p)
        return p, y


class NegRecall(object):

    def __init__(self):
        super().__init__()

    def beta_coverage_set(self, P, betas):
        threshold = 1 - betas
        return P >= threshold

    def _compute_loss(self, P, Y, betas):
        recall = (self.beta_coverage_set(P, betas)
                  & Y.bool()).float().sum(dim=-1) / Y.float().sum(dim=-1)
        return 1 - recall

    def __call__(self, PXY, betas, is_torch=False, betas_is_torch=False):
        import torch
        import numpy as np
        (P, _), Y = PXY

        if not betas_is_torch:
            if isinstance(betas, np.ndarray):
                betas = torch.tensor(betas).reshape(-1, 1)
            else:
                betas = torch.as_tensor(betas).reshape(-1, 1)

        # Either a batch of P or a batch of betas, but not both
        assert len(P.shape) == 1 or (P.shape[0] == 1) or betas.shape[0] == 1, (
            P.shape, betas.shape)
        if len(P.shape) == 1:
            P = P.reshape(1, -1)

        if is_torch:
            res = self._compute_loss(P, Y, betas)
        else:
            res = self._compute_loss(torch.Tensor(P), torch.Tensor(Y),
                                     betas).numpy()
        if betas.shape[0] != 1:
            return res.reshape(-1)
        else:
            return res
