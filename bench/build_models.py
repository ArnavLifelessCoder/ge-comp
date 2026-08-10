"""Generate a corpus of CLEAN models with architecture diversity.

We train each briefly on a synthetic-but-learnable task so the weights are
genuinely trained (trained weights have different mantissa statistics than fresh
init, and calibrating on fresh init would be cheating). Everything is offline,
CPU, and deterministic per seed.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file


def _teacher_task(in_dim, out_dim, n, seed):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(in_dim, out_dim, generator=g)
    X = torch.randn(n, in_dim, generator=g)
    logits = X @ W
    y = logits.argmax(dim=1)
    return X, y


class MLP(nn.Module):
    def __init__(self, in_dim=64, hidden=256, out_dim=10, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, ch=3, classes=10, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(ch, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Linear(width * 2 * 16, classes)

    def forward(self, x):
        z = self.features(x)
        return self.head(z.flatten(1))


def _train_mlp(model, seed, steps=200):
    X, y = _teacher_task(64, 10, 2048, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(model(X), y)
        loss.backward()
        opt.step()
    return model


def _train_cnn(model, seed, steps=120):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(256, 3, 16, 16, generator=g)
    # learnable target: sign of a fixed random projection of the mean
    W = torch.randn(3 * 16 * 16, 10, generator=g)
    y = (X.flatten(1) @ W).argmax(1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(model(X), y)
        loss.backward()
        opt.step()
    return model


def build_corpus(out_dir: str, n_per_arch: int = 6, start_seed: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    idx = 0
    specs = []
    # architecture diversity for the baseline
    for i in range(n_per_arch):
        specs.append(("mlp", dict(hidden=256, depth=3), start_seed + i))
        specs.append(("mlp_wide", dict(hidden=512, depth=2), start_seed + 100 + i))
        specs.append(("cnn", dict(width=32), start_seed + 200 + i))
        specs.append(("cnn_wide", dict(width=48), start_seed + 300 + i))
    for kind, kw, seed in specs:
        torch.manual_seed(seed)
        if kind.startswith("mlp"):
            m = MLP(hidden=kw["hidden"], depth=kw["depth"])
            _train_mlp(m, seed)
            arch = kind
        else:
            m = SmallCNN(width=kw["width"])
            _train_cnn(m, seed)
            arch = kind
        sd = {k: v.contiguous() for k, v in m.state_dict().items()}
        p = os.path.join(out_dir, f"clean_{arch}_{seed}.safetensors")
        save_file(sd, p, metadata={"arch": arch, "seed": str(seed), "label": "clean"})
        paths.append((p, arch))
        idx += 1
    return paths


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/clean"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    ps = build_corpus(out, n)
    print(f"built {len(ps)} clean models in {out}")
    for p, a in ps[:8]:
        print("  ", os.path.basename(p))
