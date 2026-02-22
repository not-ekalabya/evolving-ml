# Graph Models: Evolutionary MNIST with Primitive Tensor Graphs

This project evolves neural network architectures on MNIST using a graph of primitive tensor-operation nodes. The core idea is to keep the model as a directed acyclic graph of simple ops (MatMul, Activation, Reshape, Add, Concat, etc.) and let an evolutionary algorithm mutate and recombine those graphs over generations.

The main focus is the **evolutionary algorithm**. The `single_graph_rl.py` script is an experimental, single-graph reinforcement-learning variant.

## Key Ideas

- **Primitive node graph**: Models are built from nodes like `MatMulNode`, `ActivationNode`, `ReshapeNode`, `AddNode`, and `ConcatNode`. This keeps the graph explicit and extensible.
- **Evolutionary search**: A population of models is trained briefly, evaluated, then selected, mutated, and crossed over to form the next generation.
- **DirectML GPU**: By default the code uses DirectML (Radeon GPUs) via `torch_directml`. If DirectML is not available, it will raise an error.

## Evolutionary Algorithm (main.py)

**Starting point**
- Minimal model: `input -> flatten -> MatMul(784->10) -> output`

**Mutation operators**
- `insert_layer`: insert MatMul + Activation before the final classifier
- `change_width`: change hidden width and repair downstream MatMul
- `change_activation`: switch activation type
- `add_skip`: add residual connection via AddNode (with projection if needed)
- `add_concat`: concatenate two branches, then project to target width

**Crossover**
- Aligns nodes by type and compatible shapes
- Interpolates MatMul weights: `W_child = alpha * W_A + (1 - alpha) * W_B`
- Non-matching nodes are inherited from a random parent

**Selection**
- Models are evaluated on a small validation subset each generation
- The best models are kept as elites
- Invalid graphs are discarded

**Efficiency tweaks**
- Small population size
- Short training bursts with early stop
- Cheap evaluation most generations, full eval periodically

**Final training**
- At the end, the best valid model is trained on the full training set for a few epochs

## Running

### Evolutionary search

```bash
python main.py
```

### Real-time plots

```bash
python visual.py
```

### Experimental single-graph RL

```bash
python single_graph_rl.py
```

## Notes

- `single_graph_rl.py` is experimental and intentionally separate from the main evolutionary algorithm.
- The code is designed to keep the **graph structure explicit** so you can extend to other architectures and domains.
- Some DirectML operators may fall back internally to CPU; this is a backend limitation.
