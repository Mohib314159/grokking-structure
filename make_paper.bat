"""
test_equivalence.py -- is the batched trainer the same ALGORITHM as a plain
single model trained with stock torch.optim.AdamW?

    python test_equivalence.py

WHY THIS TEST LOOKS THE WAY IT DOES
  An earlier version compared parameters after 300 full-batch steps and
  demanded agreement to 2e-4. That test is invalid, and its failure mode is
  instructive: it also fails at weight decay ZERO, where there is no decay to
  order and the two code paths are performing literally the same arithmetic.

  The reason is that float32 round-off differs between `mm` and `bmm` (different
  reduction orders), and a ReLU network amplifies it. Once a single unit lands
  on the opposite side of its kink, the two trajectories separate and never
  come back. After 300 steps the divergence is 1e-1; after 50 it is 1e-7. That
  is chaos in the forward map, not a difference in the update rule.

  So this file tests the update rule directly:

    TEST 1  the gradient, in float64, at identical parameters
    TEST 2  one full AdamW step, in float64, from identical state
    TEST 3  ten steps, float64, still from identical state

  and separately REPORTS the float32 trajectory divergence as a diagnostic,
  without asserting on it.

  What the algorithm claim actually is:
    (a) decoupled decay is applied AFTER the gradient is taken at theta_t, as
        torch.optim.AdamW does, not before the forward pass;
    (b) the loss is multiplied by the run count so each run's gradient equals
        what it would be if that run were trained alone (otherwise
        cross_entropy averages over R*N examples and Adam's eps becomes an
        effective R*1e-8).

Exit code is nonzero on failure, so this is usable in CI.
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import core

TOL_GRAD = 1e-10
TOL_STEP = 1e-10


class Reference(nn.Module):
    def __init__(self, n, width, AB0, W20):
        super().__init__()
        self.W1 = nn.Parameter(AB0.T.clone())
        self.W2 = nn.Parameter(W20.clone())
        self.n = n

    def forward(self, a, b):
        x = torch.zeros(len(a), 2 * self.n, dtype=self.W1.dtype)
        x[torch.arange(len(a)), a] = 1.0
        x[torch.arange(len(a)), b + self.n] = 1.0
        return torch.relu(x @ self.W1.T) @ self.W2.T


def build(group, wd, seed, frac, width, lr, n_other, dtype):
    runs = [dict(group=group, split_seed=1000 + seed, init_seed=2000 + seed, wd=wd)]
    for k in range(n_other):
        runs.append(dict(group=group, split_seed=1050 + k, init_seed=2050 + k,
                         wd=[0.0, 0.1, 3.0][k % 3]))
    B = core.Batch(runs, frac=frac, width=width, lr=lr,
                   device=torch.device("cpu"), dtype=dtype)
    at = B.Str[0, :, :B.n].argmax(-1)
    bt = B.Str[0, :, B.n:].argmax(-1)
    ref = Reference(B.n, width, B.AB[0].detach().clone(), B.W2[0].detach().clone())
    ref = ref.to(dtype)
    opt = torch.optim.AdamW(ref.parameters(), lr=lr, betas=(0.9, 0.98),
                            eps=1e-8, weight_decay=wd)
    return B, ref, opt, at, bt


def run(group="Q8xZ3", wd=1.0, steps=10, seed=0, frac=0.70, width=128,
        lr=2e-3, n_other=5, dtype=torch.float64):
    torch.set_num_threads(1)
    B, ref, opt, at, bt = build(group, wd, seed, frac, width, lr, n_other, dtype)
    yt = B.yt[0]

    # TEST 1: gradients at identical parameters
    loss = F.cross_entropy(ref(at, bt), yt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gref = (ref.W1.grad.detach().clone(), ref.W2.grad.detach().clone())
    Bloss = F.cross_entropy(
        torch.bmm(torch.relu(torch.bmm(B.Str, B.AB)),
                  B.W2.transpose(1, 2)).reshape(-1, B.n),
        B.yt.reshape(-1)) * B.R
    B.opt.zero_grad(set_to_none=True)
    Bloss.backward()
    dg = max((B.AB.grad[0].T - gref[0]).abs().max().item(),
             (B.W2.grad[0] - gref[1]).abs().max().item())
    B.opt.zero_grad(set_to_none=True)

    # TESTS 2-3: n steps from identical state
    B, ref, opt, at, bt = build(group, wd, seed, frac, width, lr, n_other, dtype)
    yt = B.yt[0]
    d1 = None
    for t in range(steps):
        loss = F.cross_entropy(ref(at, bt), yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        B.step()
        d = max((ref.W1.detach() - B.AB[0].detach().T).abs().max().item(),
                (ref.W2.detach() - B.W2[0].detach()).abs().max().item())
        if t == 0:
            d1 = d
    return dg, d1, d


def main():
    print("core.Batch  vs  a single nn.Module + stock torch.optim.AdamW")
    print(f"float64, tolerances: gradient {TOL_GRAD:.0e}, step {TOL_STEP:.0e}\n")
    print(f"{'group':8s} {'wd':>5s} {'max |dgrad|':>12s} {'after 1 step':>13s} "
          f"{'after 10 steps':>15s}  verdict")
    bad = 0
    for group in ("Q8xZ3", "D8xZ3"):
        for wd in (0.0, 0.1, 1.0, 3.0):
            dg, d1, d10 = run(group=group, wd=wd, steps=10)
            ok = dg < TOL_GRAD and d1 < TOL_STEP and d10 < TOL_STEP * 100
            bad += not ok
            print(f"{group:8s} {wd:5g} {dg:12.2e} {d1:13.2e} {d10:15.2e}  "
                  f"{'ok' if ok else 'FAIL'}")
    if bad:
        print(f"\n{bad} configuration(s) disagree on the update rule itself. "
              f"Do not trust results from this trainer.")
        sys.exit(1)
    print("\nThe update rule matches stock AdamW: same gradient at the same "
          "parameters,\nsame step, same ten steps, in exact-enough arithmetic.")

    print("\nDIAGNOSTIC (not an assertion): float32 trajectory divergence.")
    print("ReLU networks amplify round-off. `mm` and `bmm` reduce in different")
    print("orders, so the two code paths separate once any unit crosses its")
    print("kink. Note that wd = 0 diverges too, where there is no decay to")
    print("order and the arithmetic is identical -- which is the proof that")
    print("this is round-off and not an algorithmic difference.")
    print(f"\n{'group':8s} {'wd':>5s} {'50 steps':>11s} {'300 steps':>11s}")
    for group in ("Q8xZ3",):
        for wd in (0.0, 1.0):
            _, _, d50 = run(group=group, wd=wd, steps=50, dtype=torch.float32)
            _, _, d300 = run(group=group, wd=wd, steps=300, dtype=torch.float32)
            print(f"{group:8s} {wd:5g} {d50:11.2e} {d300:11.2e}")


if __name__ == "__main__":
    main()
