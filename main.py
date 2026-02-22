# Graph-Based AI Architecture for MNIST Classification
# Using HuggingFace datasets library

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset

def get_device(strict=True):
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        if strict:
            raise RuntimeError("torch_directml not available; cannot use Radeon GPU.")
        return "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Primitive Node Definitions
# =========================

class PrimitiveNode(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.inputs = []

    def add_input(self, node_name):
        self.inputs.append(node_name)

    def forward(self, tensors):
        raise NotImplementedError


class MatMulNode(PrimitiveNode):

    def __init__(self, name, in_features, out_features, bias=True):
        super().__init__(name)
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        if bias:
            self.b = nn.Parameter(torch.zeros(out_features))
        else:
            self.b = None

    def forward(self, tensors):
        x = tensors[self.inputs[0]]
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y


class AddNode(PrimitiveNode):

    def __init__(self, name):
        super().__init__(name)

    def forward(self, tensors):
        if len(self.inputs) != 2:
            raise ValueError(f"AddNode '{self.name}' expects exactly 2 inputs.")
        return tensors[self.inputs[0]] + tensors[self.inputs[1]]


class ConcatNode(PrimitiveNode):

    def __init__(self, name, dim=1):
        super().__init__(name)
        self.dim = dim

    def forward(self, tensors):
        if len(self.inputs) < 2:
            raise ValueError(f"ConcatNode '{self.name}' expects at least 2 inputs.")
        return torch.cat([tensors[n] for n in self.inputs], dim=self.dim)


class ActivationNode(PrimitiveNode):

    def __init__(self, name, activation_type):
        super().__init__(name)
        self.activation_type = activation_type.lower()
        if self.activation_type not in {"relu", "gelu", "sigmoid", "tanh"}:
            raise ValueError(f"Unsupported activation: {activation_type}")

    def forward(self, tensors):
        x = tensors[self.inputs[0]]
        if self.activation_type == "relu":
            return torch.relu(x)
        if self.activation_type == "gelu":
            return torch.nn.functional.gelu(x)
        if self.activation_type == "sigmoid":
            return torch.sigmoid(x)
        return torch.tanh(x)


class ReshapeNode(PrimitiveNode):

    def __init__(self, name, operation, parameters=None):
        super().__init__(name)
        self.operation = operation
        self.parameters = parameters

    def forward(self, tensors):
        x = tensors[self.inputs[0]]
        if self.operation == "flatten":
            return x.reshape(x.shape[0], -1)
        if self.operation == "reshape":
            return x.reshape(*self.parameters)
        if self.operation == "permute":
            return x.permute(*self.parameters)
        raise ValueError(f"Unsupported reshape operation: {self.operation}")


class ReduceNode(PrimitiveNode):

    def __init__(self, name, operation, dim=None):
        super().__init__(name)
        self.operation = operation
        self.dim = dim

    def forward(self, tensors):
        x = tensors[self.inputs[0]]
        if self.operation == "mean":
            return x.mean(dim=self.dim)
        if self.operation == "sum":
            return x.sum(dim=self.dim)
        if self.operation == "max":
            return x.max(dim=self.dim).values
        raise ValueError(f"Unsupported reduce operation: {self.operation}")


class IndexNode(PrimitiveNode):

    def __init__(self, name, mode, parameters=None):
        super().__init__(name)
        self.mode = mode
        self.parameters = parameters

    def forward(self, tensors):
        x = tensors[self.inputs[0]]
        if self.mode == "gather":
            dim, index = self.parameters
            return torch.gather(x, dim, index)
        if self.mode == "mask":
            mask = self.parameters
            return x[mask]
        raise ValueError(f"Unsupported index mode: {self.mode}")


# =========================
# Graph Model
# =========================

class GraphModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.nodes = nn.ModuleDict()
        self.execution_order = []

    def add_node(self, node):
        self.nodes[node.name] = node
        self.execution_order.append(node.name)

    def forward(self, x):

        tensors = {}
        tensors["input"] = x

        for name in self.execution_order:
            node = self.nodes[name]
            tensors[name] = node(tensors)

        return tensors[self.execution_order[-1]]


def _is_compatible(node_a, node_b):
    if type(node_a) is not type(node_b):
        return False
    if isinstance(node_a, MatMulNode):
        return node_a.W.shape == node_b.W.shape and (node_a.b is None) == (node_b.b is None)
    if isinstance(node_a, ActivationNode):
        return node_a.activation_type == node_b.activation_type
    if isinstance(node_a, ReshapeNode):
        return node_a.operation == node_b.operation and node_a.parameters == node_b.parameters
    if isinstance(node_a, ReduceNode):
        return node_a.operation == node_b.operation and node_a.dim == node_b.dim
    if isinstance(node_a, IndexNode):
        return node_a.mode == node_b.mode and node_a.parameters == node_b.parameters
    if isinstance(node_a, AddNode):
        return True
    if isinstance(node_a, ConcatNode):
        return node_a.dim == node_b.dim
    return False


def _clone_node_from(parent_node):
    if isinstance(parent_node, MatMulNode):
        out_features = parent_node.W.shape[1]
        in_features = parent_node.W.shape[0]
        child = MatMulNode(parent_node.name, in_features, out_features, bias=parent_node.b is not None)
        with torch.no_grad():
            child.W.copy_(parent_node.W)
            if child.b is not None:
                child.b.copy_(parent_node.b)
        return child
    if isinstance(parent_node, ActivationNode):
        return ActivationNode(parent_node.name, parent_node.activation_type)
    if isinstance(parent_node, ReshapeNode):
        return ReshapeNode(parent_node.name, parent_node.operation, parent_node.parameters)
    if isinstance(parent_node, ReduceNode):
        return ReduceNode(parent_node.name, parent_node.operation, parent_node.dim)
    if isinstance(parent_node, IndexNode):
        return IndexNode(parent_node.name, parent_node.mode, parent_node.parameters)
    if isinstance(parent_node, AddNode):
        return AddNode(parent_node.name)
    if isinstance(parent_node, ConcatNode):
        return ConcatNode(parent_node.name, dim=parent_node.dim)
    raise ValueError(f"Unsupported node type for cloning: {type(parent_node)}")


def _crossover_matmul(node_a, node_b):
    out_features = node_a.W.shape[1]
    in_features = node_a.W.shape[0]
    child = MatMulNode(node_a.name, in_features, out_features, bias=node_a.b is not None)
    alpha = torch.rand(1).item()
    with torch.no_grad():
        child.W.copy_(alpha * node_a.W + (1.0 - alpha) * node_b.W)
        if child.b is not None:
            child.b.copy_(alpha * node_a.b + (1.0 - alpha) * node_b.b)
    return child


def _crossover_activation(node_a, node_b):
    if node_a.activation_type == node_b.activation_type:
        return ActivationNode(node_a.name, node_a.activation_type)
    chosen = node_a if torch.rand(1).item() < 0.5 else node_b
    return ActivationNode(node_a.name, chosen.activation_type)


def crossover_models(parent_a, parent_b):
    if parent_a.execution_order != parent_b.execution_order:
        raise ValueError("Parent graphs must have identical execution order to preserve topology.")

    child = GraphModel()

    for name in parent_a.execution_order:
        node_a = parent_a.nodes[name]
        node_b = parent_b.nodes.get(name, None)

        if node_b is not None and _is_compatible(node_a, node_b):
            if isinstance(node_a, MatMulNode):
                child_node = _crossover_matmul(node_a, node_b)
            elif isinstance(node_a, ActivationNode):
                child_node = _crossover_activation(node_a, node_b)
            else:
                chosen = node_a if torch.rand(1).item() < 0.5 else node_b
                child_node = _clone_node_from(chosen)
        else:
            chosen = node_a if node_b is None or torch.rand(1).item() < 0.5 else node_b
            child_node = _clone_node_from(chosen)

        child_node.inputs = list(node_a.inputs)
        child.add_node(child_node)

    return child


# =========================
# Evolution Utilities
# =========================

def build_minimal_model():

    model = GraphModel()

    flatten = ReshapeNode("flatten", "flatten", None)
    flatten.add_input("input")
    model.add_node(flatten)

    fc_out = MatMulNode("fc_out", 784, 10, bias=True)
    fc_out.add_input("flatten")
    model.add_node(fc_out)

    return model


def forward_with_tensors(model, x):
    tensors = {"input": x}
    for name in model.execution_order:
        tensors[name] = model.nodes[name](tensors)
    return tensors


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def describe_architecture(model):
    parts = []
    for name in model.execution_order:
        node = model.nodes[name]
        if isinstance(node, ReshapeNode):
            parts.append(f"{name}:Reshape({node.operation})")
        elif isinstance(node, MatMulNode):
            parts.append(f"{name}:MatMul({node.W.shape[0]}->{node.W.shape[1]})")
        elif isinstance(node, ActivationNode):
            parts.append(f"{name}:Act({node.activation_type})")
        elif isinstance(node, AddNode):
            parts.append(f"{name}:Add")
        elif isinstance(node, ConcatNode):
            parts.append(f"{name}:Concat(dim={node.dim})")
        elif isinstance(node, ReduceNode):
            parts.append(f"{name}:Reduce({node.operation})")
        elif isinstance(node, IndexNode):
            parts.append(f"{name}:Index({node.mode})")
        else:
            parts.append(f"{name}:{type(node).__name__}")
    return " -> ".join(parts)


def clone_model(parent):
    child = GraphModel()
    for name in parent.execution_order:
        node = parent.nodes[name]
        child_node = _clone_node_from(node)
        child_node.inputs = list(node.inputs)
        child.add_node(child_node)
    return child


def architecture_signature(model):
    return "|".join(
        [
            f"{type(model.nodes[n]).__name__}:{getattr(model.nodes[n], 'activation_type', '')}"
            f":{getattr(model.nodes[n], 'operation', '')}"
            f":{getattr(model.nodes[n], 'mode', '')}"
            f":{getattr(model.nodes[n], 'dim', '')}"
            f":{getattr(model.nodes[n], 'W', None).shape if isinstance(model.nodes[n], MatMulNode) else ''}"
            for n in model.execution_order
        ]
    )


def novelty_score(model, elite_signatures):
    if not elite_signatures:
        return 0.0
    sig = architecture_signature(model)
    def dist(a, b):
        # simple distance: proportion of non-matching tokens
        ta = a.split("|")
        tb = b.split("|")
        m = min(len(ta), len(tb))
        mismatch = sum(1 for i in range(m) if ta[i] != tb[i])
        mismatch += abs(len(ta) - len(tb))
        return mismatch / max(1, max(len(ta), len(tb)))
    dists = [dist(sig, s) for s in elite_signatures]
    return sum(dists) / len(dists)


def _find_compatible_node(target, candidates, used):
    for name, node in candidates.items():
        if name in used:
            continue
        if _is_compatible(target, node):
            return name, node
    return None, None


def crossover_models_flexible(parent_a, parent_b):
    # Preserve topology of parent A, align compatible nodes from parent B
    child = GraphModel()
    used_b = set()

    for name in parent_a.execution_order:
        node_a = parent_a.nodes[name]
        node_b = parent_b.nodes[name] if name in parent_b.nodes else None

        if node_b is None or not _is_compatible(node_a, node_b):
            _, node_b = _find_compatible_node(node_a, parent_b.nodes, used_b)

        if node_b is not None and _is_compatible(node_a, node_b):
            used_b.add(node_b.name)
            if isinstance(node_a, MatMulNode):
                child_node = _crossover_matmul(node_a, node_b)
            elif isinstance(node_a, ActivationNode):
                child_node = _crossover_activation(node_a, node_b)
            else:
                chosen = node_a if torch.rand(1).item() < 0.5 else node_b
                child_node = _clone_node_from(chosen)
        else:
            chosen = node_a if torch.rand(1).item() < 0.5 else (parent_b.nodes[name] if name in parent_b.nodes else node_a)
            child_node = _clone_node_from(chosen)

        child_node.inputs = list(node_a.inputs)
        child.add_node(child_node)

    return child


def mutate_model(model, rng, force=None):
    # Mutations operate on a cloned model
    child = clone_model(model)

    # Use a dummy forward to infer shapes
    dummy = torch.zeros(2, 1, 28, 28)
    try:
        tensors = forward_with_tensors(child, dummy)
    except Exception:
        return child

    act_nodes = [n for n in child.execution_order if isinstance(child.nodes[n], ActivationNode)]
    matmul_nodes = [n for n in child.execution_order if isinstance(child.nodes[n], MatMulNode)]
    has_2d_nodes = any(t.ndim == 2 for t in tensors.values())

    options = []
    if matmul_nodes:
        options.append("insert_layer")
    if act_nodes:
        options.append("change_activation")
    if len(matmul_nodes) >= 3:
        options.append("change_width")
    if has_2d_nodes and len(child.execution_order) >= 3:
        options.append("add_skip")
    if has_2d_nodes and len(child.execution_order) >= 3:
        options.append("add_concat")

    if not options:
        return child

    # Bias toward growth when the graph is shallow
    if force in options:
        mutation_type = force
    elif len(matmul_nodes) <= 1 and "insert_layer" in options:
        mutation_type = "insert_layer" if rng.rand() < 0.8 else rng.choice(options)
    else:
        mutation_type = rng.choice(options)

    if mutation_type == "change_activation":
        if act_nodes:
            name = rng.choice(act_nodes)
            node = child.nodes[name]
            new_act = rng.choice(["relu", "gelu", "sigmoid", "tanh"])
            node.activation_type = new_act
        return child

    if mutation_type == "insert_layer":
        # Insert a MatMul + Activation before the final MatMul output
        if not matmul_nodes:
            return child
        last_mm = matmul_nodes[-1]
        last_idx = child.execution_order.index(last_mm)
        if last_idx == 0:
            return child
        prev_name = child.execution_order[last_idx - 1]
        prev_tensor = tensors[prev_name]
        if prev_tensor.ndim != 2:
            return child
        in_features = prev_tensor.shape[1]
        new_width = int(rng.choice([32, 64, 128, 192, 256]))

        mm_name = f"mm_ins_{rng.randint(0, 10**6)}"
        act_name = f"act_ins_{rng.randint(0, 10**6)}"

        mm = MatMulNode(mm_name, in_features, new_width, bias=True)
        mm.add_input(prev_name)
        act = ActivationNode(act_name, rng.choice(["relu", "gelu", "sigmoid", "tanh"]))
        act.add_input(mm_name)

        # Replace last MatMul to accept new width
        last_node = child.nodes[last_mm]
        new_last = MatMulNode(last_mm, new_width, last_node.W.shape[1], bias=last_node.b is not None)
        new_last.inputs = list(last_node.inputs)

        child.nodes[last_mm] = new_last

        # Rewire last MatMul input to activation
        new_last.inputs = [act_name]

        # Insert into execution order
        child.execution_order = (
            child.execution_order[:last_idx] + [mm_name, act_name] + child.execution_order[last_idx:]
        )
        child.nodes[mm_name] = mm
        child.nodes[act_name] = act
        return child

    if mutation_type == "change_width":
        # Pick a MatMul that isn't the final output and adjust width with downstream fix
        if len(matmul_nodes) < 3:
            return child
        idx = rng.randint(0, len(matmul_nodes) - 1)
        mm_name = matmul_nodes[idx]
        next_mm_name = matmul_nodes[idx + 1]

        mm_node = child.nodes[mm_name]
        next_node = child.nodes[next_mm_name]

        new_width = int(rng.choice([32, 64, 128, 192, 256]))

        new_mm = MatMulNode(mm_name, mm_node.W.shape[0], new_width, bias=mm_node.b is not None)
        new_mm.inputs = list(mm_node.inputs)
        child.nodes[mm_name] = new_mm

        new_next = MatMulNode(next_mm_name, new_width, next_node.W.shape[1], bias=next_node.b is not None)
        new_next.inputs = list(next_node.inputs)
        child.nodes[next_mm_name] = new_next

        return child

    if mutation_type == "add_skip":
        # Add skip connection via AddNode, with projection if needed
        if len(child.execution_order) < 3:
            return child

        src_idx = rng.randint(0, len(child.execution_order) - 3)
        tgt_idx = rng.randint(src_idx + 1, len(child.execution_order) - 2)
        src_name = child.execution_order[src_idx]
        tgt_name = child.execution_order[tgt_idx]

        src_tensor = tensors[src_name]
        tgt_tensor = tensors[tgt_name]

        if src_tensor.ndim != 2 or tgt_tensor.ndim != 2:
            return child

        add_name = f"add_{rng.randint(0, 10**6)}"
        proj_name = f"proj_{rng.randint(0, 10**6)}"

        if src_tensor.shape[1] != tgt_tensor.shape[1]:
            proj = MatMulNode(proj_name, src_tensor.shape[1], tgt_tensor.shape[1], bias=False)
            proj.add_input(src_name)
            child.nodes[proj_name] = proj
            proj_input = proj_name
            insert_after = tgt_idx
            insert_nodes = [proj_name, add_name]
        else:
            proj_input = src_name
            insert_after = tgt_idx
            insert_nodes = [add_name]

        add = AddNode(add_name)
        add.inputs = [tgt_name, proj_input]
        child.nodes[add_name] = add

        # Rewire downstream nodes that used tgt_name to use add_name
        for name in child.execution_order[tgt_idx + 1:]:
            node = child.nodes[name]
            node.inputs = [add_name if x == tgt_name else x for x in node.inputs]

        # Insert new nodes after target
        child.execution_order = (
            child.execution_order[:insert_after + 1] + insert_nodes + child.execution_order[insert_after + 1:]
        )

        return child

    if mutation_type == "add_concat":
        if len(child.execution_order) < 3:
            return child

        idxs = [i for i, n in enumerate(child.execution_order) if tensors[n].ndim == 2]
        if len(idxs) < 2:
            return child

        src_idx1 = rng.choice(idxs)
        src_idx2 = rng.choice([i for i in idxs if i != src_idx1])
        if src_idx1 > src_idx2:
            src_idx1, src_idx2 = src_idx2, src_idx1

        tgt_candidates = [
            i
            for i in idxs
            if i > src_idx2 and len(child.nodes[child.execution_order[i]].inputs) == 1
        ]
        if not tgt_candidates:
            return child
        tgt_idx = rng.choice(tgt_candidates)

        src1 = child.execution_order[src_idx1]
        src2 = child.execution_order[src_idx2]
        tgt_name = child.execution_order[tgt_idx]

        src1_tensor = tensors[src1]
        src2_tensor = tensors[src2]
        tgt_tensor = tensors[tgt_name]

        if src1_tensor.shape[0] != src2_tensor.shape[0]:
            return child

        concat_name = f"concat_{rng.randint(0, 10**6)}"
        proj_name = f"proj_{rng.randint(0, 10**6)}"

        concat = ConcatNode(concat_name, dim=1)
        concat.inputs = [src1, src2]
        child.nodes[concat_name] = concat

        concat_width = src1_tensor.shape[1] + src2_tensor.shape[1]
        proj = MatMulNode(proj_name, concat_width, tgt_tensor.shape[1], bias=True)
        proj.add_input(concat_name)
        child.nodes[proj_name] = proj

        # Rewire target to take projection
        tgt_node = child.nodes[tgt_name]
        tgt_node.inputs = [proj_name]

        # Insert concat and proj before target
        child.execution_order = (
            child.execution_order[:tgt_idx] + [concat_name, proj_name] + child.execution_order[tgt_idx:]
        )

        return child

    return child


def is_valid_model(model, device):
    try:
        dummy = torch.zeros(2, 1, 28, 28, device=device)
        out = model(dummy)
        return out.ndim == 2 and out.shape[1] == 10
    except Exception:
        return False


def train_brief(model, loader, device, steps=100):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    step = 0
    for batch in loader:
        x = batch["image"].float().to(device)
        y = batch["label"].to(device)

        if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] != 1:
            x = x.permute(1, 0, 2, 3)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        if not loss.requires_grad:
            return False
        loss.backward()
        optimizer.step()

        step += 1
        if step >= steps:
            break
    return True


def evaluate_accuracy(model, loader, device, max_batches=50):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["image"].float().to(device)
            y = batch["label"].to(device)

            if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] != 1:
                x = x.permute(1, 0, 2, 3)
            if x.ndim == 3:
                x = x.unsqueeze(1)

            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            if i + 1 >= max_batches:
                break

    return correct / max(1, total)


# =========================
# Load MNIST via datasets
# =========================

dataset = load_dataset("mnist")


def transform(example):

    # Convert PIL -> numpy -> torch and normalize to [0, 1]
    image = example["image"]
    if isinstance(image, list):
        image = np.stack([np.array(im, dtype=np.float32) for im in image], axis=0)
    else:
        image = np.array(image, dtype=np.float32)

    image = torch.from_numpy(image) / 255.0

    # Ensure channel dimension exists (1, 28, 28) or (B, 1, 28, 28)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[0] != 1:
            image = image.unsqueeze(1)

    label = example["label"]
    if isinstance(label, list):
        label = torch.tensor(label, dtype=torch.long)
    else:
        label = torch.tensor(label, dtype=torch.long)

    return {"image": image, "label": label}



dataset = dataset.with_transform(transform)


train_loader = DataLoader(
    dataset["train"],
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset["test"],
    batch_size=64
)


def run_evolution(
    generations=20,
    population_size=6,
    elites=2,
    mutation_rate=0.7,
    crossover_rate=0.3,
    training_steps=100,
    val_batches=50,
    seed=42,
    on_generation=None,
):
    device = get_device(strict=True)
    rng = np.random.RandomState(seed)

    population = [build_minimal_model().to(device) for _ in range(population_size)]
    for gen in range(generations):
        scored = []

        for model in population:
            trained = train_brief(model, train_loader, device, steps=training_steps)
            params = count_parameters(model)
            if not trained:
                acc = 0.0
                fitness = -1e9
            else:
                acc = evaluate_accuracy(model, test_loader, device, max_batches=val_batches)
                fitness = acc
            scored.append((fitness, acc, params, model))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_fitness, best_acc, best_params, best_model = scored[0]
        avg_acc = sum(s[1] for s in scored) / len(scored)
        max_nodes_model = max(scored, key=lambda x: len(x[3].execution_order))[3]
        max_nodes = len(max_nodes_model.execution_order)

        print(f"Generation {gen} | Best Acc: {best_acc:.4f} | Avg Acc: {avg_acc:.4f} | Best Params: {best_params}")
        print(f"Best Nodes: {len(best_model.execution_order)} | Arch: {describe_architecture(best_model)}")
        print(f"Max Nodes: {max_nodes} | Arch: {describe_architecture(max_nodes_model)}")

        if on_generation is not None:
            on_generation(
                {
                    "generation": gen,
                    "best_acc": best_acc,
                    "avg_acc": avg_acc,
                    "best_params": best_params,
                    "best_nodes": len(best_model.execution_order),
                    "max_nodes": max_nodes,
                    "best_model": best_model,
                    "max_nodes_model": max_nodes_model,
                }
            )

        next_population = []
        for i in range(elites):
            next_population.append(clone_model(scored[i][3]).to(device))

        # Inject at least one growth mutation from the best model
        if len(next_population) < population_size:
            growth_parent = scored[0][3]
            growth_child = mutate_model(growth_parent, rng, force="insert_layer")
            growth_child.to(device)
            if is_valid_model(growth_child, device):
                next_population.append(growth_child)

        while len(next_population) < population_size:
            fallback_parent = None
            if rng.rand() < crossover_rate and len(scored) >= 2:
                parent_a = scored[rng.randint(0, len(scored))][3]
                parent_b = scored[rng.randint(0, len(scored))][3]
                fallback_parent = parent_a
                child = crossover_models_flexible(parent_a, parent_b)
            else:
                parent = scored[rng.randint(0, len(scored))][3]
                fallback_parent = parent
                child = clone_model(parent)

            if rng.rand() < mutation_rate:
                child = mutate_model(child, rng)

            child.to(device)
            if not is_valid_model(child, device):
                child = clone_model(fallback_parent).to(device)
            next_population.append(child)

        population = next_population

    best = max(population, key=lambda m: evaluate_accuracy(m, test_loader, device, max_batches=val_batches))
    final_acc = evaluate_accuracy(best, test_loader, device, max_batches=val_batches)
    print("Final Best Accuracy:", final_acc)
    print("Final Best Params:", count_parameters(best))
    print("Final Best Arch:", describe_architecture(best))
    return best


if __name__ == "__main__":
    run_evolution()
