import torch
from torch import nn
from torch.autograd import profiler

import numpy as np


class MyModule(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)

    def forward(self, x, mask):
        with profiler.record_function('Linear'):
            out = self.linear(x)

        with profiler.record_function('Mask'):
            threshold = out.sum(dim=1).mean().item()
            idx = np.argwhere(mask.cpu().numpy() > threshold)
            idx = torch.from_numpy(idx).cuda()

        return out, idx


def main():
    model = MyModule(500, 10).cuda()
    data = torch.rand(128, 500).cuda()
    mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

    model(data, mask)

    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        out, idx = model(data, mask)

    report = prof.key_averages(group_by_stack_n=5)
    report = report.table(sort_by='self_cpu_time_total', row_limit=5)

    print(report)


if __name__ == "__main__":
    main()
