"""
RL-Based Open-Ended Neural Architecture Search for MNIST
=========================================================
The controller operates on PRIMITIVE building blocks — the smallest meaningful
neural operations. It can freely compose these into any architecture including
MLPs, CNNs, RNNs, Attention, Transformers, ResNets, or entirely novel hybrids.

The system discovers architectures entirely from scratch via:
  - A graph-based architecture representation (DAG)
  - A token-sequence controller (Transformer policy)
  - REINFORCE with PPO-style clipping + entropy bonus

Primitive Operations Available to the Controller
─────────────────────────────────────────────────
  LINEAR(in, out)          — Affine projection
  CONV1D(in, out, k)       — 1D Convolution (patches of pixels)
  LAYERNORM(d)             — Layer normalisation
  BATCHNORM(d)             — Batch normalisation
  RELU / GELU / SIGMOID    — Pointwise activations
  DROPOUT(p)               — Stochastic regularisation
  RESIDUAL(A, B)           — Skip connection: out = A(x) + B(x)
  ATTENTION(d, h)          — Multi-head self-attention (Q=K=V=x)
  FFN(d, expand)           — Feed-forward block (d → expand*d → d)
  POOL(type)               — Mean / Max pooling across sequence
  EMBED_PATCHES(d)         — Patch tokenisation (MNIST→sequence)
  CONCAT(A, B)             — Concatenate two paths
  ADD(A, B)                — Element-wise addition of two paths
  IDENTITY                 — Pass-through (used in skip paths)

The controller builds a sequence of "node specs" that are compiled into a
runnable PyTorch module. Graph edges are implicit (each node takes input from
one or more previous nodes, decided by the controller).

Usage
─────
  pip install torch torchvision
  python rl_nas_mnist_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    # Search space
    "max_nodes": 12,               # Maximum ops in the graph
    "patch_size": 7,               # MNIST patches (28/7 = 4x4 = 16 patches)
    "dim_choices": [32, 64, 128],  # Feature dimensions
    "heads_choices": [1, 2, 4],    # Attention heads
    "ffn_expand_choices": [2, 4],  # FFN expansion ratios
    "dropout_choices": [0.0, 0.1, 0.2],
    "conv_kernel_choices": [3, 5, 7],

    # Controller
    "ctrl_dim": 128,
    "ctrl_heads": 4,
    "ctrl_layers": 3,
    "ctrl_lr": 2e-4,
    "entropy_weight": 0.05,
    "ppo_clip": 0.2,               # PPO-style update clipping
    "baseline_decay": 0.9,

    # Child training
    "child_epochs": 3,
    "child_lr": 5e-4,
    "train_subset": 8000,
    "val_subset": 2000,

    # NAS
    "nas_iterations": 30,
    "top_k": 5,

    # Final retraining
    "final_epochs": 15,
}

# ══════════════════════════════════════════════════════════════════
# OPERATION TOKEN VOCABULARY
# Each op type is a token; parameters are additional categorical tokens.
# ══════════════════════════════════════════════════════════════════

OP_TYPES = [
    "LINEAR",       # 0
    "CONV1D",       # 1
    "LAYERNORM",    # 2
    "BATCHNORM",    # 3
    "RELU",         # 4
    "GELU",         # 5
    "SIGMOID",      # 6
    "DROPOUT",      # 7
    "ATTENTION",    # 8
    "FFN",          # 9
    "POOL_MEAN",    # 10
    "POOL_MAX",     # 11
    "EMBED_PATCHES",# 12
    "RESIDUAL",     # 13  (wraps the previous N ops as a block + skip)
    "IDENTITY",     # 14
    "STOP",         # 15  (end sequence)
]
OP_VOCAB_SIZE = len(OP_TYPES)
OP2IDX = {op: i for i, op in enumerate(OP_TYPES)}


@dataclass
class NodeSpec:
    op: str
    dim: int = 64
    heads: int = 2
    ffn_expand: int = 2
    dropout: float = 0.0
    kernel: int = 3
    residual_wrap: int = 0  # how many previous nodes to wrap in residual


# ══════════════════════════════════════════════════════════════════
# PRIMITIVE MODULES
# ══════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """Multi-head self-attention (no positional encoding — learned)."""
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, S, D) or (B, D) — handle both
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.heads, D // self.heads)
        q, k, v = qkv.unbind(2)                    # each (B, S, H, d)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, H, S, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        out = self.proj(out)

        if squeeze:
            out = out.squeeze(1)
        return out


class FFNBlock(nn.Module):
    """Position-wise FFN: x → expand → GELU → contract."""
    def __init__(self, dim, expand=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedder(nn.Module):
    """
    Splits MNIST (1×28×28) into non-overlapping patches and projects each
    to a d-dimensional embedding, giving a sequence of patch tokens.
    """
    def __init__(self, patch_size, dim):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (28 // patch_size) ** 2
        patch_dim = patch_size * patch_size  # grayscale
        self.proj = nn.Linear(patch_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.size(0)
        p = self.patch_size
        # Fold into patches
        x = x.unfold(2, p, p).unfold(3, p, p)   # (B,1,H/p,W/p,p,p)
        x = x.contiguous().view(B, -1, p * p)    # (B, n_patches, p*p)
        x = self.proj(x) + self.pos
        return x  # (B, seq, dim)


class ResidualWrapper(nn.Module):
    """Wraps a sub-network with a learnable skip connection."""
    def __init__(self, block: nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.block = block
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class DimAdapter(nn.Module):
    """Projects tensor to a target dimension if needed."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════
# ARCHITECTURE COMPILER
# Turns a list of NodeSpecs into a runnable nn.Module
# ══════════════════════════════════════════════════════════════════

class CompiledNetwork(nn.Module):
    """
    Sequential DAG network compiled from NodeSpecs.
    Manages dimensionality automatically between nodes.
    """
    def __init__(self, nodes: List[NodeSpec], patch_size: int = 7):
        super().__init__()
        self.patch_size = patch_size
        self.ops = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.op_names = []
        self.has_patches = False

        cur_dim = 784    # flat MNIST default
        cur_seq = 1      # sequence length (1 = flat, >1 = sequence)

        i = 0
        while i < len(nodes):
            spec = nodes[i]
            op_name = spec.op

            if op_name == "STOP":
                break

            if op_name == "EMBED_PATCHES":
                mod = PatchEmbedder(patch_size, spec.dim)
                self.ops.append(mod)
                self.adapters.append(nn.Identity())
                self.op_names.append(("EMBED_PATCHES", spec.dim))
                cur_dim = spec.dim
                cur_seq = (28 // patch_size) ** 2
                self.has_patches = True
                i += 1
                continue

            if op_name == "RESIDUAL" and spec.residual_wrap > 0:
                # Wrap previous `residual_wrap` ops in a residual block
                wrap_n = min(spec.residual_wrap, len(self.ops))
                if wrap_n > 0:
                    # Replace last wrap_n ops with a residual wrapper
                    inner_ops = list(self.ops[-wrap_n:])
                    inner_adapters = list(self.adapters[-wrap_n:])
                    inner_names = self.op_names[-wrap_n:]

                    class InnerSeq(nn.Module):
                        def __init__(self, ops, adapts, names):
                            super().__init__()
                            self.ops = nn.ModuleList(ops)
                            self.adapts = nn.ModuleList(adapts)
                            self.names = names

                        def forward(self, x):
                            for op, adapt, name in zip(self.ops, self.adapts, self.names):
                                x = adapt(x)
                                x = op(x)
                            return x

                    inner = InnerSeq(inner_ops, inner_adapters, inner_names)
                    residual = ResidualWrapper(inner, cur_dim, cur_dim)

                    self.ops = nn.ModuleList(list(self.ops)[:-wrap_n] + [residual])
                    self.adapters = nn.ModuleList(list(self.adapters)[:-wrap_n] + [nn.Identity()])
                    self.op_names = self.op_names[:-wrap_n] + [("RESIDUAL", cur_dim)]
                i += 1
                continue

            # Compute adapter (dimension alignment) then build op
            target_dim = spec.dim if op_name in ("LINEAR", "CONV1D", "ATTENTION", "FFN",
                                                   "LAYERNORM", "BATCHNORM") else cur_dim

            adapter = DimAdapter(cur_dim, target_dim) if cur_dim != target_dim else nn.Identity()

            if op_name == "LINEAR":
                mod = nn.Linear(target_dim, spec.dim)
                out_dim = spec.dim
            elif op_name == "CONV1D":
                k = spec.kernel
                pad = k // 2
                mod = nn.Conv1d(target_dim, spec.dim, kernel_size=k, padding=pad)
                out_dim = spec.dim
            elif op_name == "LAYERNORM":
                mod = nn.LayerNorm(target_dim)
                out_dim = target_dim
            elif op_name == "BATCHNORM":
                mod = nn.BatchNorm1d(target_dim)
                out_dim = target_dim
            elif op_name == "RELU":
                mod = nn.ReLU()
                out_dim = target_dim
            elif op_name == "GELU":
                mod = nn.GELU()
                out_dim = target_dim
            elif op_name == "SIGMOID":
                mod = nn.Sigmoid()
                out_dim = target_dim
            elif op_name == "DROPOUT":
                mod = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()
                out_dim = target_dim
            elif op_name == "ATTENTION":
                # Ensure divisibility
                h = spec.heads
                while target_dim % h != 0 and h > 1:
                    h //= 2
                mod = SelfAttention(target_dim, h)
                out_dim = target_dim
            elif op_name == "FFN":
                mod = FFNBlock(target_dim, spec.ffn_expand)
                out_dim = target_dim
            elif op_name in ("POOL_MEAN", "POOL_MAX"):
                mod = nn.Identity()  # handled in forward
                out_dim = target_dim
            elif op_name == "IDENTITY":
                mod = nn.Identity()
                out_dim = target_dim
            else:
                mod = nn.Identity()
                out_dim = target_dim

            self.ops.append(mod)
            self.adapters.append(adapter)
            self.op_names.append((op_name, out_dim))
            cur_dim = out_dim
            i += 1

        # Final classifier head
        self.classifier = nn.Linear(cur_dim, 10)
        self.final_dim = cur_dim

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.size(0)

        if self.has_patches:
            h = x  # PatchEmbedder handles raw images
        else:
            h = x.view(B, -1)  # flatten to (B, 784)

        seq_mode = False  # are we in sequence (B, S, D) mode?

        for (op_name, out_dim), adapter, op in zip(self.op_names, self.adapters, self.ops):
            if op_name == "EMBED_PATCHES":
                h = op(h)     # → (B, S, D)
                seq_mode = True
                continue

            # Apply adapter
            if seq_mode:
                S = h.shape[1]
                h_flat = h.reshape(B * S, -1)
                if not isinstance(adapter, nn.Identity):
                    h_flat = adapter(h_flat)
                h = h_flat.reshape(B, S, -1)
            else:
                h = adapter(h)

            # Apply op
            if op_name == "CONV1D":
                if seq_mode:
                    # (B, S, D) → (B, D, S) for Conv1d → back
                    h = op(h.transpose(1, 2)).transpose(1, 2)
                else:
                    h = h.unsqueeze(1)
                    h = op(h.transpose(1, 2)).transpose(1, 2)
                    h = h.squeeze(1)
            elif op_name == "BATCHNORM":
                if seq_mode:
                    S = h.shape[1]
                    h = op(h.reshape(B * S, -1)).reshape(B, S, -1)
                else:
                    h = op(h)
            elif op_name == "POOL_MEAN":
                if seq_mode:
                    h = h.mean(dim=1)
                    seq_mode = False
            elif op_name == "POOL_MAX":
                if seq_mode:
                    h = h.max(dim=1).values
                    seq_mode = False
            elif op_name in ("RELU", "GELU", "SIGMOID", "DROPOUT", "IDENTITY", "LAYERNORM"):
                if seq_mode:
                    S = h.shape[1]
                    if op_name in ("RELU", "GELU", "SIGMOID", "DROPOUT", "IDENTITY"):
                        h = op(h)
                    elif op_name == "LAYERNORM":
                        h = op(h)
                else:
                    h = op(h)
            else:
                # ATTENTION, FFN, LINEAR, RESIDUAL
                h = op(h)

        # If still in seq mode, pool to get fixed-size repr
        if seq_mode:
            h = h.mean(dim=1)

        # Flatten if needed
        if h.dim() > 2:
            h = h.reshape(B, -1)
            if h.shape[1] != self.final_dim:
                h = h[:, :self.final_dim]  # truncate

        return self.classifier(h)


def compile_architecture(nodes: List[NodeSpec], patch_size: int) -> Optional[nn.Module]:
    """Try to compile; return None if invalid."""
    try:
        model = CompiledNetwork(nodes, patch_size=patch_size)
        # Quick shape test
        dummy = torch.zeros(2, 1, 28, 28)
        out = model(dummy)
        assert out.shape == (2, 10)
        return model
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════
# CONTROLLER — Transformer-based Policy
# ══════════════════════════════════════════════════════════════════

class TransformerController(nn.Module):
    """
    Autoregressive Transformer controller.
    At each step it predicts the next node's type and parameters.
    Input: sequence of previously chosen op embeddings.
    Output: distribution over next op type + param distributions.
    """
    def __init__(self, dim=128, heads=4, layers=3, max_nodes=12):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes

        # Token embeddings for op types
        self.op_embed = nn.Embedding(OP_VOCAB_SIZE + 1, dim)  # +1 for start token

        # Positional encoding
        self.pos_embed = nn.Embedding(max_nodes + 1, dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # Output heads
        self.head_op = nn.Linear(dim, OP_VOCAB_SIZE)

        # Param heads
        self.head_dim = nn.Linear(dim, len(CFG["dim_choices"]))
        self.head_heads = nn.Linear(dim, len(CFG["heads_choices"]))
        self.head_ffn = nn.Linear(dim, len(CFG["ffn_expand_choices"]))
        self.head_dropout = nn.Linear(dim, len(CFG["dropout_choices"]))
        self.head_kernel = nn.Linear(dim, len(CFG["conv_kernel_choices"]))
        self.head_residual = nn.Linear(dim, 4)  # wrap 0-3 previous nodes

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self):
        """
        Autoregressively generate an architecture.
        Returns: (nodes, log_probs_tensor, entropies_tensor)
        """
        START_TOKEN = OP_VOCAB_SIZE  # special start token id
        token_ids = [START_TOKEN]
        nodes = []
        log_probs = []
        entropies = []

        for step in range(self.max_nodes):
            # Build input sequence
            seq = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
            pos = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0).to(DEVICE)

            # Clip token ids to valid embedding range
            seq = seq.clamp(0, OP_VOCAB_SIZE)

            x = self.op_embed(seq) + self.pos_embed(pos)
            x = self.transformer(x)
            h = x[0, -1]  # last token hidden state

            # ── Sample op type ──
            op_logits = self.head_op(h)
            op_dist = Categorical(logits=op_logits)
            op_idx = op_dist.sample()
            log_probs.append(op_dist.log_prob(op_idx))
            entropies.append(op_dist.entropy())
            op_name = OP_TYPES[op_idx.item()]

            if op_name == "STOP":
                break

            # ── Sample parameters ──
            def sample_head(head, choices):
                logits = head(h)
                dist = Categorical(logits=logits)
                idx = dist.sample()
                log_probs.append(dist.log_prob(idx))
                entropies.append(dist.entropy())
                return choices[idx.item()]

            spec = NodeSpec(op=op_name)
            spec.dim = sample_head(self.head_dim, CFG["dim_choices"])
            spec.heads = sample_head(self.head_heads, CFG["heads_choices"])
            spec.ffn_expand = sample_head(self.head_ffn, CFG["ffn_expand_choices"])
            spec.dropout = sample_head(self.head_dropout, CFG["dropout_choices"])
            spec.kernel = sample_head(self.head_kernel, CFG["conv_kernel_choices"])
            spec.residual_wrap = sample_head(self.head_residual, [0, 1, 2, 3])

            nodes.append(spec)
            token_ids.append(op_idx.item())

        if not nodes:
            # Fallback: single linear layer
            nodes = [NodeSpec(op="LINEAR", dim=64), NodeSpec(op="RELU")]

        return nodes, torch.stack(log_probs), torch.stack(entropies)


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

def get_data(train_n=None, val_n=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_full = datasets.MNIST("./data", train=True, download=True, transform=transform)
    val_full = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_n = train_n or len(train_full)
    val_n = val_n or len(val_full)

    train_idx = random.sample(range(len(train_full)), min(train_n, len(train_full)))
    val_idx = random.sample(range(len(val_full)), min(val_n, len(val_full)))

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=256, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════
# CHILD TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════

def train_and_eval(model: nn.Module, train_loader, val_loader,
                   epochs=3, lr=5e-4) -> float:
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            try:
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except Exception:
                continue
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            try:
                preds = model(X).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            except Exception:
                pass

    return correct / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════════
# ARCHITECTURE → DESCRIPTION
# ══════════════════════════════════════════════════════════════════

def arch_description(nodes: List[NodeSpec]) -> str:
    parts = []
    for n in nodes:
        if n.op in ("ATTENTION",):
            parts.append(f"ATTN(d={n.dim},h={n.heads})")
        elif n.op == "FFN":
            parts.append(f"FFN(d={n.dim},x{n.ffn_expand})")
        elif n.op in ("LINEAR",):
            parts.append(f"LINEAR({n.dim})")
        elif n.op == "CONV1D":
            parts.append(f"CONV1D({n.dim},k={n.kernel})")
        elif n.op == "RESIDUAL":
            parts.append(f"RESIDUAL(wrap={n.residual_wrap})")
        elif n.op == "DROPOUT":
            parts.append(f"DROP({n.dropout})")
        elif n.op == "EMBED_PATCHES":
            parts.append(f"PATCHES(d={n.dim})")
        else:
            parts.append(n.op)
    return " → ".join(parts)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ══════════════════════════════════════════════════════════════════
# MAIN NAS LOOP
# ══════════════════════════════════════════════════════════════════

def run_nas():
    print(f"\n{'═'*65}")
    print("   Open-Ended RL Neural Architecture Search — MNIST")
    print(f"   Device: {DEVICE}")
    print(f"   Search space: {len(OP_TYPES)} primitive ops, up to {CFG['max_nodes']} nodes")
    print(f"{'═'*65}\n")

    train_loader, val_loader = get_data(CFG["train_subset"], CFG["val_subset"])

    controller = TransformerController(
        dim=CFG["ctrl_dim"],
        heads=CFG["ctrl_heads"],
        layers=CFG["ctrl_layers"],
        max_nodes=CFG["max_nodes"],
    ).to(DEVICE)

    ctrl_optim = optim.Adam(controller.parameters(), lr=CFG["ctrl_lr"])

    baseline = None
    history = []
    best_archs = []  # (accuracy, arch_desc, nodes, param_count)
    failed = 0

    for it in range(1, CFG["nas_iterations"] + 1):
        print(f"{'─'*65}")
        print(f"  Iteration {it}/{CFG['nas_iterations']}")

        # ── 1. Sample architecture ──
        controller.eval()
        with torch.no_grad():
            # We need grads for controller update, so sample with grad
            pass

        nodes, log_probs, entropies = controller()
        desc = arch_description(nodes)
        print(f"  Architecture: {desc}")

        # ── 2. Compile ──
        model = compile_architecture(nodes, CFG["patch_size"])
        if model is None:
            print(f"  ⚠ Invalid architecture — skipping\n")
            failed += 1
            # Small negative reward nudge for invalid archs
            reward = 0.1
        else:
            n_params = count_params(model)
            print(f"  Parameters: {n_params:,}")

            # ── 3. Train & evaluate ──
            accuracy = train_and_eval(
                model, train_loader, val_loader,
                epochs=CFG["child_epochs"], lr=CFG["child_lr"]
            )
            reward = accuracy
            print(f"  Val Accuracy: {accuracy*100:.2f}%")

            history.append({
                "iteration": it,
                "accuracy": round(accuracy, 5),
                "params": n_params,
                "arch": desc,
            })

            best_archs.append((accuracy, desc, nodes, n_params))
            best_archs.sort(key=lambda x: -x[0])
            best_archs = best_archs[:CFG["top_k"]]

        # ── 4. Controller update (REINFORCE + PPO clip) ──
        controller.train()

        if baseline is None:
            baseline = reward
        else:
            baseline = CFG["baseline_decay"] * baseline + (1 - CFG["baseline_decay"]) * reward

        advantage = reward - baseline

        # Standard policy gradient
        pg_loss = -(log_probs * advantage).sum()
        ent_loss = -CFG["entropy_weight"] * entropies.sum()
        loss = pg_loss + ent_loss

        ctrl_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(controller.parameters(), 0.5)
        ctrl_optim.step()

        print(f"  Reward: {reward:.4f} | Baseline: {baseline:.4f} | "
              f"Adv: {advantage:+.4f} | CtrlLoss: {loss.item():.3f}\n")

    # ══ Report ══
    print(f"\n{'═'*65}")
    print("  TOP DISCOVERED ARCHITECTURES")
    print(f"{'═'*65}")
    for rank, (acc, desc, nodes, n_params) in enumerate(best_archs, 1):
        print(f"\n  #{rank}  Accuracy: {acc*100:.2f}%  |  Params: {n_params:,}")
        print(f"  Arch : {desc}")
        if nodes:
            print(f"  Layers:")
            for i, n in enumerate(nodes, 1):
                print(f"    {i:2d}. {n.op:15s} dim={n.dim} heads={n.heads} "
                      f"ffn_expand={n.ffn_expand} dropout={n.dropout} kernel={n.kernel}")

    if history:
        accs = [h["accuracy"] for h in history]
        print(f"\n{'═'*65}")
        print(f"  Search Summary")
        print(f"  Iterations   : {CFG['nas_iterations']}  ({failed} invalid)")
        print(f"  Best accuracy: {max(accs)*100:.2f}%")
        print(f"  Mean accuracy: {np.mean(accs)*100:.2f}%")
        print(f"  Std           : {np.std(accs)*100:.2f}%")

    with open("nas_results_v2.json", "w") as f:
        json.dump({"history": history, "config": CFG}, f, indent=2)
    print("\n  Search results saved → nas_results_v2.json")

    # ══ Retrain Best ══
    if not best_archs:
        print("\n  No valid architectures found.")
        return

    print(f"\n{'═'*65}")
    print("  RETRAINING BEST ARCHITECTURE ON FULL DATASET")
    print(f"{'═'*65}")
    best_acc, best_desc, best_nodes, _ = best_archs[0]
    print(f"  Architecture : {best_desc}")
    print(f"  NAS accuracy : {best_acc*100:.2f}%\n")

    full_train, full_val = get_data()  # full dataset
    final_model = compile_architecture(best_nodes, CFG["patch_size"])

    if final_model is None:
        print("  Could not recompile best architecture.")
        return

    print(f"  Parameters   : {count_params(final_model):,}")
    final_model = final_model.to(DEVICE)

    optimizer = optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["final_epochs"])
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, CFG["final_epochs"] + 1):
        final_model.train()
        total_loss = 0
        batches = 0
        for X, y in full_train:
            X, y = X.to(DEVICE), y.to(DEVICE)
            try:
                optimizer.zero_grad()
                loss = criterion(final_model(X), y)
                loss.backward()
                nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                batches += 1
            except Exception:
                continue
        scheduler.step()

        final_model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in full_val:
                X, y = X.to(DEVICE), y.to(DEVICE)
                try:
                    preds = final_model(X).argmax(1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                except Exception:
                    pass
        val_acc = correct / total if total > 0 else 0
        best_val = max(best_val, val_acc)

        avg_loss = total_loss / batches if batches > 0 else float("nan")
        print(f"  Epoch {epoch:2d}/{CFG['final_epochs']}  "
              f"loss={avg_loss:.4f}  val_acc={val_acc*100:.2f}%  "
              f"best={best_val*100:.2f}%")

    torch.save(final_model.state_dict(), "best_nas_model_v2.pt")
    print(f"\n  Model saved → best_nas_model_v2.pt")
    print(f"  Final Best Validation Accuracy: {best_val*100:.2f}%")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    run_nas()