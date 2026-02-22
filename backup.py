# Graph-Based AI Architecture for MNIST Classification
# Using HuggingFace datasets library

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset


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


# =========================
# Build MNIST Graph
# =========================

model = GraphModel()

flatten = ReshapeNode("flatten", "flatten", None)
flatten.add_input("input")
model.add_node(flatten)

fc1 = MatMulNode("fc1", 784, 128, bias=True)
fc1.add_input("flatten")
model.add_node(fc1)

relu = ActivationNode("relu", "relu")
relu.add_input("fc1")
model.add_node(relu)

fc2 = MatMulNode("fc2", 128, 10, bias=True)
fc2.add_input("relu")
model.add_node(fc2)


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


# =========================
# Training Setup
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# =========================
# Training Loop
# =========================

for epoch in range(5):

    model.train()
    total_loss = 0

    for batch in train_loader:

        x = batch["image"].float().to(device)
        y = batch["label"].to(device)

        # Ensure channel dimension exists
        if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] != 1:
            x = x.permute(1, 0, 2, 3)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.2f}")


# =========================
# Evaluation
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in test_loader:

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

print("Test Accuracy:", correct / total)
