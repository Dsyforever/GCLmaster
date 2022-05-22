import torch


class noise(nn.Module):
    def __init__(self,Q):
        super().__init__()
        self.Q=Q

    def forward(self, xb):
        Aug_xb = xb + self.Q * 2 * (torch.randn(xb.shape) - 0.5).to(xb.device)
        return Aug_xb