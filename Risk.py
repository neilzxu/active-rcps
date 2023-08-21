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
        """Get mark array that indicates the labeled that are covered
        :param w_P: n x classes tensor that's label cost times predicted prob. for label
        :param idx_arr: n x classes tensor that class indices sorted by highest to lowest predicted prob.
        :param top_idx: n len tensor of number of indices covered under a choice of beta
        """
        import torch
        mask = torch.zeros(idx_arr.shape).bool()
        row_indices = torch.cat([torch.full((top_idx[i].item() + 1,), i) for i in range(len(top_idx))])
        col_indices = torch.cat([idx_arr[i, :(top_idx[i] + 1)] for i in range(len(top_idx))])
        mask[row_indices, col_indices] = True
        return mask
    
    def beta_coverage_set(self, P, betas):
        import torch
        w_P = P * self.torch_w.reshape(1, -1)
        sort_arr, idx_arr = torch.sort(w_P, descending=True, dim=-1, stable=True)
        sums_arr = torch.cumsum(sort_arr, dim=-1)
        top_idx = torch.argmax((sums_arr >= betas).int(), dim=-1)
        
        # if all sums are below beta, set to last possible index (i.e., include all labels)
        sum_small_idxs = torch.all(sums_arr < betas, dim=-1)
        top_idx[sum_small_idxs] = self.torch_w.shape[0] - 1

        return self._coverage_mask(idx_arr, top_idx)

    def _compute_loss(self, P, Y, betas):
        import torch
        w_P = P * self.torch_w.reshape(1, -1)
        sort_arr, idx_arr = torch.sort(w_P, descending=True, dim=-1, stable=True)
        sums_arr = torch.cumsum(sort_arr, dim=-1)
        top_idx = torch.argmax((sums_arr >= betas).int(), dim=-1)
        
        # if all sums are below beta, set to last possible index (i.e., include all labels)
        sum_small_idxs = torch.all(sums_arr < betas, dim=-1)
        top_idx[sum_small_idxs] = self.torch_w.shape[0] - 1
        top_idx = top_idx.reshape(-1, 1)
        
        label_top_idx = torch.argmax((idx_arr == Y.reshape(-1, 1)).int(), dim=-1).reshape(-1, 1)
        miscover_arr = (label_top_idx > top_idx).float()
        assert torch.all(Y <= 1000), Y
        return miscover_arr * self.torch_w[Y].reshape(-1)
    
    def __call__(self, PXY, betas, is_torch=False, betas_is_torch=False):
        """Either beta can be single or multipel rows (i.e., single beta)"""
        
        import torch
        (P, _), Y = PXY
        
        betas = (betas if betas_is_torch else torch.Tensor([betas])).reshape(-1, 1)
        
        # Either a batch of P or a batch of betas, but not both
        assert len(P.shape) == 1 or (P.shape[0] == 1) or betas.shape[0] == 1, (P.shape, betas.shape)
        if len(P.shape) == 1:
            P = P.reshape(1, -1)
        
        if is_torch:
            res = self._compute_loss(P, Y, betas)
        else:
            res = self._compute_loss(torch.Tensor(P), torch.Tensor([Y]).reshape(-1).int(), betas).numpy()
            
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
            return (1. - self.beta_coverage_set(P, betas)[np.arange(P.shape[0]), Y].float()) * self.torch_w[Y]
        else:
            return (1. - self.beta_coverage_set(P, betas)[np.arange(P.shape[0]), Y]) * self.w[Y]
        
    def sample(self):
        import torch
        
        p = torch.rand(1)
        y = torch.bernoulli(p)
        return p, y

        
